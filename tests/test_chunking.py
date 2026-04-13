"""
Unit and integration tests for the chunking layer.

Unit tests cover pure functions with no external dependencies.
Integration tests run the full processor pipeline with a mock embedding service
so no model or Elasticsearch is needed.
"""

import email as email_lib
import json
from io import BytesIO

from langchain_core.documents import Document

from src.ingestion.email_processor import (
    EmailProcessor,
    _extract_thread_id,
    _strip_quoted_content,
)
from src.ingestion.json_processor import (
    JSONProcessor,
    _chunk_text,
    _detect_structure,
    _flatten,
)

# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------

def test_flatten_long_text_goes_to_content():
    content, _ = _flatten({"clause": "This clause governs the termination of the agreement between all parties involved herein"})
    assert any("clause" in c for c in content)

def test_flatten_short_text_goes_to_metadata():
    _, meta = _flatten({"status": "APPROVED"})
    assert meta.get("status") == "APPROVED"

def test_flatten_short_text_not_in_content():
    content, _ = _flatten({"status": "APPROVED"})
    assert not content

def test_flatten_number_goes_to_metadata():
    _, meta = _flatten({"amount": 50000})
    assert meta.get("amount") == 50000

def test_flatten_list_with_no_long_text_is_skipped():
    content, meta = _flatten({"ids": ["abc-001", "abc-002", "abc-003"]})
    assert not content

def test_flatten_nested_dict_recurses():
    data = {"outer": {"inner": "This nested value is definitely long enough to exceed the ten word minimum"}}
    content, _ = _flatten(data)
    assert any("inner" in c for c in content)

def test_flatten_list_containing_long_text_is_included():
    data = {"notes": ["Short", "This note is long enough to pass the ten word threshold for content fields"]}
    content, _ = _flatten(data)
    assert any("notes" in c for c in content)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_empty_returns_empty():
    assert _chunk_text([]) == []

def test_chunk_text_short_input_single_chunk():
    chunks = _chunk_text(["word " * 50])
    assert len(chunks) == 1

def test_chunk_text_long_input_splits():
    # 500 words > CHUNK_SIZE (400), expect 2 chunks
    chunks = _chunk_text([" ".join(["word"] * 500)])
    assert len(chunks) == 2

def test_chunk_text_overlap_present():
    # Second chunk should start with words from the tail of the first
    chunks = _chunk_text([" ".join(["word"] * 500)])
    first_tail = set(chunks[0].split()[-60:])
    second_head = set(chunks[1].split()[:60])
    assert first_tail & second_head  # overlap exists


# ---------------------------------------------------------------------------
# _detect_structure
# ---------------------------------------------------------------------------

def test_detect_structure_identifies_array():
    assert _detect_structure(BytesIO(b'[{"key": "val"}]')) == "array"

def test_detect_structure_identifies_object():
    assert _detect_structure(BytesIO(b'{"key": "val"}')) == "object"

def test_detect_structure_with_leading_whitespace():
    assert _detect_structure(BytesIO(b'   [1, 2, 3]')) == "array"

def test_detect_structure_seeks_back_to_zero():
    f = BytesIO(b'[1, 2, 3]')
    _detect_structure(f)
    assert f.tell() == 0


# ---------------------------------------------------------------------------
# Email pure functions
# ---------------------------------------------------------------------------

def test_strip_quoted_removes_gt_lines():
    body = "My reply.\n\n> Quoted line one\n> Quoted line two\n\nMore of my reply."
    result = _strip_quoted_content(body)
    assert "Quoted line" not in result
    assert "My reply" in result

def test_strip_quoted_cuts_at_reply_header():
    body = "My answer.\n\nOn Mon, 1 Jan 2024, Alice wrote:\nOriginal message text here."
    result = _strip_quoted_content(body)
    assert "Original message" not in result
    assert "My answer" in result

def test_extract_thread_id_uses_first_reference():
    raw = "From: a@b.com\nReferences: <root@mail.com> <child@mail.com>\nMessage-ID: <msg@mail.com>\n\nbody"
    msg = email_lib.message_from_string(raw)
    assert _extract_thread_id(msg) == "root@mail.com"

def test_extract_thread_id_falls_back_to_in_reply_to():
    raw = "From: a@b.com\nIn-Reply-To: <parent@mail.com>\nMessage-ID: <msg@mail.com>\n\nbody"
    msg = email_lib.message_from_string(raw)
    assert _extract_thread_id(msg) == "parent@mail.com"

def test_extract_thread_id_falls_back_to_message_id():
    raw = "From: a@b.com\nMessage-ID: <standalone@mail.com>\n\nbody"
    msg = email_lib.message_from_string(raw)
    assert _extract_thread_id(msg) == "standalone@mail.com"


# ---------------------------------------------------------------------------
# EmailProcessor semantic chunking
# ---------------------------------------------------------------------------

LONG_EML = """\
From: alice@lawfirm.com
To: bob@corp.com
Subject: Detailed Contract Terms
Message-ID: <long-001@lawfirm.com>
Date: Mon, 01 Jan 2024 10:00:00 +0000
Content-Type: text/plain; charset=utf-8

{body}
"""

MULTI_TOPIC_BODY = """\
The indemnification clause requires each party to hold the other harmless from \
any third-party claims arising out of the performance of this agreement. \
Indemnification obligations survive the termination of this contract for a period of five years.

Payment terms stipulate that all invoices are due within thirty days of receipt. \
Late payments shall accrue interest at a rate of one point five percent per month. \
Wire transfer details are provided in Schedule B attached hereto.

Governing law for this agreement shall be the State of New York. \
Any disputes shall be resolved exclusively through binding arbitration under AAA rules. \
The prevailing party shall be entitled to recover reasonable legal fees and costs.
"""


def test_email_processor_chunks_are_documents(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "multi.eml"
    eml_file.write_text(LONG_EML.format(body=MULTI_TOPIC_BODY))
    docs, embeddings = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert all(isinstance(d, Document) for d in docs)
    assert len(docs) == len(embeddings)


def test_email_processor_chunks_preserve_metadata(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "multi.eml"
    eml_file.write_text(LONG_EML.format(body=MULTI_TOPIC_BODY))
    docs, _ = EmailProcessor(mock_embedding_service).process(str(eml_file))
    for doc in docs:
        assert doc.metadata["sender"] == "alice@lawfirm.com"
        assert doc.metadata["subject"] == "Detailed Contract Terms"


def test_email_processor_no_empty_chunks(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "multi.eml"
    eml_file.write_text(LONG_EML.format(body=MULTI_TOPIC_BODY))
    docs, _ = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert all(d.page_content.strip() for d in docs)


# ---------------------------------------------------------------------------
# Processor integration (no ES, mock embeddings)
# ---------------------------------------------------------------------------

EML = """\
From: alice@lawfirm.com
To: bob@corp.com
Subject: Contract renewal terms
Message-ID: <test-001@lawfirm.com>
Date: Mon, 01 Jan 2024 10:00:00 +0000
Content-Type: text/plain; charset=utf-8

We need to discuss the renewal of the contract before the deadline next month.
Please review the attached terms and confirm the penalty clauses are acceptable.
"""


def test_email_processor_eml_returns_one_doc(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "test.eml"
    eml_file.write_text(EML)
    docs, embeddings = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert len(docs) == 1
    assert len(embeddings) == 1


def test_email_processor_eml_metadata(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "test.eml"
    eml_file.write_text(EML)
    docs, _ = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert docs[0].metadata["sender"] == "alice@lawfirm.com"
    assert docs[0].metadata["subject"] == "Contract renewal terms"


def test_email_processor_eml_subject_in_content(tmp_path, mock_embedding_service):
    eml_file = tmp_path / "test.eml"
    eml_file.write_text(EML)
    docs, _ = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert "Contract renewal terms" in docs[0].page_content


def test_email_processor_empty_body_returns_empty(tmp_path, mock_embedding_service):
    eml = "From: a@b.com\nMessage-ID: <x@y.com>\nDate: Mon, 01 Jan 2024 10:00:00 +0000\n\n"
    eml_file = tmp_path / "empty.eml"
    eml_file.write_text(eml)
    docs, embeddings = EmailProcessor(mock_embedding_service).process(str(eml_file))
    assert docs == []
    assert embeddings == []


def test_json_processor_array_produces_docs(mock_embedding_service):
    data = [
        {"id": "1", "description": "This clause governs the termination of the agreement between parties upon written notice."},
        {"id": "2", "description": "The penalty for breach of contract shall not exceed the total value of the agreement."},
    ]
    docs, embeddings = JSONProcessor(mock_embedding_service).process(BytesIO(json.dumps(data).encode()), "test.json")
    assert len(docs) > 0
    assert len(docs) == len(embeddings)


def test_json_processor_array_no_content_returns_empty(mock_embedding_service):
    data = [{"id": "abc", "code": "XYZ", "flag": True}]
    docs, embeddings = JSONProcessor(mock_embedding_service).process(BytesIO(json.dumps(data).encode()), "test.json")
    assert docs == []
    assert embeddings == []


def test_json_processor_object_produces_docs(mock_embedding_service):
    data = {
        "title": "Service Agreement",
        "body": "This agreement sets forth the terms and conditions under which the provider agrees to deliver services to the client.",
        "status": "ACTIVE",
    }
    docs, embeddings = JSONProcessor(mock_embedding_service).process(BytesIO(json.dumps(data).encode()), "test.json")
    assert len(docs) > 0
    assert len(docs) == len(embeddings)


def test_json_processor_object_short_fields_not_in_content(mock_embedding_service):
    data = {
        "body": "This agreement sets forth the terms and conditions under which the provider agrees to deliver services.",
        "status": "ACTIVE",
    }
    docs, _ = JSONProcessor(mock_embedding_service).process(BytesIO(json.dumps(data).encode()), "test.json")
    combined = " ".join(d.page_content for d in docs)
    assert "ACTIVE" not in combined
