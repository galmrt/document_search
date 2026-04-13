"""
Tests for the retrieval layer.

Unit tests cover the RRF merge logic and filter wiring using a mocked ES client.
Integration tests (marked with @pytest.mark.integration) require Elasticsearch
and index a small set of known documents to verify mechanics.

Run all:        pytest tests/test_retrieval.py -v
Skip ES tests:  pytest tests/test_retrieval.py -v -m "not integration"
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.utils.es_service import ESService

TEST_FILE_ID = "retrieval-test-fixture-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(doc_id: str, file_name: str, doc_type: str = "pdf") -> dict:
    return {
        "_id": doc_id,
        "fields": {
            "content": ["some content"],
            "file_name": [file_name],
            "doc_type": [doc_type],
            "page_number": [1],
            "sender": [],
            "subject": [],
            "email_date": [],
        },
    }


@pytest.fixture
def es_svc():
    """ESService with a mocked Elasticsearch client — no real ES needed."""
    with patch("elasticsearch.Elasticsearch"):
        svc = ESService("http://localhost:9200")
    return svc


# ---------------------------------------------------------------------------
# RRF merge logic
# ---------------------------------------------------------------------------

def test_rrf_doc_in_both_lists_ranks_first(es_svc):
    """
    doc_c at rank 2 in both BM25 and KNN should beat doc_a/doc_b
    which each appear at rank 1 in only one list.

    doc_a score: 1/(60+1)        ≈ 0.0164
    doc_b score: 1/(60+1)        ≈ 0.0164
    doc_c score: 1/(60+2)*2      ≈ 0.0323  ← wins
    """
    es_svc.es.search = Mock(side_effect=[
        {"hits": {"hits": [_hit("id_a", "doc_a.pdf"), _hit("id_c", "doc_c.pdf")]}},  # BM25
        {"hits": {"hits": [_hit("id_b", "doc_b.pdf"), _hit("id_c", "doc_c.pdf")]}},  # KNN
    ])
    results = es_svc.search("test query", [0.1] * 1024, size=3)
    assert results[0]["file_name"] == "doc_c.pdf"


def test_rrf_returns_correct_count(es_svc):
    es_svc.es.search = Mock(side_effect=[
        {"hits": {"hits": [_hit("id_a", "a.pdf"), _hit("id_b", "b.pdf"), _hit("id_c", "c.pdf")]}},
        {"hits": {"hits": [_hit("id_d", "d.pdf"), _hit("id_e", "e.pdf")]}},
    ])
    results = es_svc.search("test", [0.1] * 1024, size=2)
    assert len(results) == 2


def test_rrf_single_result_list(es_svc):
    """When KNN returns nothing, BM25 results still surface."""
    es_svc.es.search = Mock(side_effect=[
        {"hits": {"hits": [_hit("id_a", "only.pdf")]}},
        {"hits": {"hits": []}},
    ])
    results = es_svc.search("test", [0.1] * 1024, size=5)
    assert len(results) == 1
    assert results[0]["file_name"] == "only.pdf"


# ---------------------------------------------------------------------------
# Filter wiring
# ---------------------------------------------------------------------------

def test_filter_adds_bool_to_bm25_query(es_svc):
    es_svc.es.search = Mock(return_value={"hits": {"hits": []}})
    es_svc.search("test", [0.1] * 1024, size=5, doc_type="email")

    bm25_query = es_svc.es.search.call_args_list[0].kwargs["query"]
    assert "bool" in bm25_query
    assert bm25_query["bool"]["filter"] == {"term": {"doc_type": "email"}}


def test_filter_adds_filter_to_knn(es_svc):
    es_svc.es.search = Mock(return_value={"hits": {"hits": []}})
    es_svc.search("test", [0.1] * 1024, size=5, doc_type="pdf")

    knn = es_svc.es.search.call_args_list[1].kwargs["knn"]
    assert knn["filter"] == {"term": {"doc_type": "pdf"}}


def test_no_filter_uses_plain_match_query(es_svc):
    es_svc.es.search = Mock(return_value={"hits": {"hits": []}})
    es_svc.search("test", [0.1] * 1024, size=5, doc_type=None)

    bm25_query = es_svc.es.search.call_args_list[0].kwargs["query"]
    assert "match" in bm25_query
    assert "bool" not in bm25_query


def test_filter_value_is_lowercased(es_svc):
    es_svc.es.search = Mock(return_value={"hits": {"hits": []}})
    es_svc.search("test", [0.1] * 1024, size=5, doc_type="PDF")

    bm25_query = es_svc.es.search.call_args_list[0].kwargs["query"]
    assert bm25_query["bool"]["filter"] == {"term": {"doc_type": "pdf"}}


# ---------------------------------------------------------------------------
# ES integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def es_with_data(es_service):
    pdf_doc = Document(
        page_content="Force majeure clauses excuse a party from performance obligations due to extraordinary and unforeseeable events.",
        metadata={"page_number": 1},
    )
    email_doc = Document(
        page_content="Payment deadlines and penalty fees apply when invoices remain outstanding beyond thirty calendar days.",
        metadata={
            "email_id": "eid-test-001",
            "thread_id": "tid-test-001",
            "sender": "billing@corp.com",
            "email_date": "2024-01-01T00:00:00+00:00",
            "subject": "Payment reminder",
        },
    )

    es_service.index_chunks("test_fixture.pdf", TEST_FILE_ID, 1, [pdf_doc], [[0.1] * 1024])
    es_service.index_emails("test_fixture.mbox", TEST_FILE_ID + "-email", [email_doc], [[0.2] * 1024])
    es_service.es.indices.refresh(index="knowledge_base")

    yield es_service

    es_service.es.delete_by_query(
        index="knowledge_base",
        body={"query": {"terms": {"file_id": [TEST_FILE_ID, TEST_FILE_ID + "-email"]}}},
    )


@pytest.mark.integration
def test_doc_type_filter_excludes_other_type(es_with_data):
    results = es_with_data.search(
        "force majeure obligations",
        [0.1] * 1024,
        size=10,
        doc_type="email",
    )
    assert all(r["doc_type"] == "email" for r in results)


@pytest.mark.integration
def test_doc_type_filter_pdf_excludes_email(es_with_data):
    results = es_with_data.search(
        "payment penalty deadlines",
        [0.2] * 1024,
        size=10,
        doc_type="pdf",
    )
    assert all(r["doc_type"] == "pdf" for r in results)


@pytest.mark.integration
def test_bm25_finds_lexical_match(es_with_data):
    results = es_with_data.search(
        "force majeure",
        [0.1] * 1024,
        size=10,
    )
    file_names = [r["file_name"] for r in results]
    assert "test_fixture.pdf" in file_names
