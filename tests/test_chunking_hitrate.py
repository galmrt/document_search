"""
Hit-rate test for chunking quality against 06-ex-02-rev-0323-a11y.pdf
(California DOT right-of-way legal deed templates, 92 chunks).

Each question is tied to a specific page and unique keywords from that page.
A question is a HIT only if the expected page appears in the top-K results.

Run with:
    pytest tests/test_chunking_hitrate.py -v -s

Requires Elasticsearch to be running and the document to be indexed.
"""

import os
import pytest
from src.utils.embedding_service import EmbeddingService
from src.utils.es_service import ESService

TOP_K = 5
TARGET_FILE = "Incarceration or E-Carceration_ California?s SB 10 Bail Reform an.pdf"

# (question, expected_page, keywords_that_confirm_correct_chunk)
QUESTIONS = [
    (
        "Who was Kenneth Humphrey and what happened to him?",
        3,
        ["seven dollars", "bottle of cologne", "350,000"],
    ),
    (
        "Who created the first commercial bail bonding business in America?",
        4,
        ["McDonough", "1896", "San Francisco"],
    ),
    (
        "How does pretrial detention affect conviction rates?",
        6,
        ["25%", "43%", "convicted"],
    ),
    (
        "What was the Bail Reform Act of 1966?",
        6,
        ["Bail Reform Act of 1966", "statutory right", "financial status"],
    ),
    (
        "What did United States v. Salerno decide about preventative detention?",
        8,
        ["Salerno", "regulatory", "penal"],
    ),
    (
        "What is California's average bail amount?",
        11,
        ["$50,000", "five times", "national average"],
    ),
    (
        "What did New Jersey's Criminal Justice Reform Act achieve?",
        10,
        ["New Jersey", "CJRA", "pretrial jail population"],
    ),
    (
        "What are the due process challenges to SB 10?",
        13,
        ["Due Process", "Fifth", "Fourteenth Amendment"],
    ),
    (
        "What does the Eighth Amendment say about bail?",
        15,
        ["Eighth Amendment", "excessive bail"],
    ),
    (
        "What is PAS and what does it do under SB 10?",
        23,
        ["Pretrial Assessment Services", "PAS", "risk assessment"],
    ),
    (
        "Why are risk assessments in SB 10 potentially discriminatory?",
        26,
        ["discriminatory", "risk assessment", "criminal justice system"],
    ),
    (
        "What socioeconomic factors do risk assessments consider?",
        29,
        ["age at time of first arrest", "socioeconomic", "custody status"],
    ),
]


@pytest.fixture(scope="module")
def services():
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es = ESService(es_url)
    try:
        es.es.cluster.health(timeout="2s")
    except Exception:
        pytest.skip("Elasticsearch not available")

    count = es.es.count(index="knowledge_base", body={"query": {"term": {"file_name": TARGET_FILE}}})
    if count["count"] == 0:
        pytest.skip(f"{TARGET_FILE} not indexed")

    return es, EmbeddingService()


def retrieve(es_service, embedding_service, query: str, k: int) -> list[dict]:
    embedding = embedding_service.encode([query])[0]
    body = {
        "field": "embedding",
        "query_vector": embedding,
        "k": k,
        "num_candidates": 100,
        "filter": {"term": {"file_name": TARGET_FILE}},
    }
    response = es_service.es.search(
        index="knowledge_base",
        source=False,
        fields=["content", "page_number"],
        knn=body,
    )
    return [
        {
            "content": hit["fields"]["content"][0],
            "page": hit["fields"]["page_number"][0],
        }
        for hit in response["hits"]["hits"]
    ]


def test_hit_rate(services):
    es_service, embedding_service = services
    hits = 0

    for question, expected_page, keywords in QUESTIONS:
        results = retrieve(es_service, embedding_service, question, TOP_K)
        pages = [r["page"] for r in results]
        combined_content = " ".join(r["content"] for r in results).lower()

        page_hit = any(
            (isinstance(p, list) and expected_page in p) or p == expected_page
            for p in pages
        )
        keyword_hit = any(kw.lower() in combined_content for kw in keywords)
        matched = page_hit and keyword_hit

        status = "HIT " if matched else "MISS"
        print(f"  [{status}] {question}")
        if not matched:
            print(f"          expected page {expected_page}, got pages {pages}")
            print(f"          keywords {keywords}")
        if matched:
            hits += 1

    hit_rate = hits / len(QUESTIONS)
    print(f"\n  Hit rate: {hits}/{len(QUESTIONS)} = {hit_rate:.0%}")
    assert hit_rate >= 0.7, f"Hit rate {hit_rate:.0%} is below 70% threshold"
