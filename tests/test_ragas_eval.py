"""
RAGAS evaluation of chunking/retrieval quality.

Target: "Incarceration or E-Carceration_ California?s SB 10 Bail Reform an.pdf"
        (Cornell Law paper on California SB 10 bail reform)

Metrics:
  - context_precision: are retrieved chunks actually relevant to the query?
  - context_recall: are all relevant chunks being retrieved?
    (uses ground_truth answers to judge whether key facts appear in contexts)

Setup:
    pip install ragas langchain-groq
    export GROQ_API_KEY=...  (free tier at console.groq.com)

    Index the document first:
        curl -X POST http://localhost:8000/upload \\
          -F "file=@data/Incarceration or E-Carceration_ California?s SB 10 Bail Reform an.pdf"

Run:
    pytest tests/test_ragas_eval.py -v -s
"""

import os
import pytest

TARGET_FILE = "Incarceration or E-Carceration_ California?s SB 10 Bail Reform an.pdf"
TOP_K = 5

# (question, ground_truth_answer)
# Ground truth covers the key facts Claude should find in the retrieved chunks.
EVAL_SET = [
    (
        "Who was Kenneth Humphrey and what was the outcome of his case?",
        "Kenneth Humphrey was arrested after stealing seven dollars and a bottle of cologne. "
        "A court ordered him held on $350,000 bail, which he could not afford. "
        "The California Court of Appeal declared California's money-bail system unconstitutional "
        "for penalizing the poor.",
    ),
    (
        "Who created the first commercial bail bonding business in America?",
        "The McDonough brothers created America's first bail bonding business in 1896 in San Francisco, "
        "by charging a fee for posting bail money as a favor to lawyers.",
    ),
    (
        "What were the negative effects of pretrial detention on defendants?",
        "Detained defendants are 25% more likely to be convicted and 43% more likely to receive "
        "jail sentences compared to defendants who are released pretrial.",
    ),
    (
        "What did the Bail Reform Act of 1984 change about bail?",
        "The Bail Reform Act of 1984 allowed courts to impose conditions of release to ensure "
        "community safety and added a rebuttable presumption of preventative detention for serious "
        "crimes like violent crimes or serious drug crimes.",
    ),
    (
        "What was the holding in United States v. Salerno?",
        "The Salerno Court upheld the constitutionality of the Bail Reform Act of 1984, ruling that "
        "preventative detention must be regulatory not penal, and is constitutional as long as there "
        "are robust procedural safeguards protecting due process rights.",
    ),
    (
        "What did New Jersey's Criminal Justice Reform Act do to bail?",
        "New Jersey's Criminal Justice Reform Act instituted pretrial service agencies to conduct "
        "risk assessments, encouraged nonmonetary release, and allowed detention without bail for "
        "dangerous defendants. Since its implementation, judicial officers nearly eliminated monetary bail "
        "and the pretrial jail population decreased by 35% between 2015 and 2017.",
    ),
    (
        "What is California's average bail amount compared to the national average?",
        "California's average bail is $50,000, more than five times the national average.",
    ),
    (
        "What are the due process challenges to SB 10?",
        "SB 10 faces due process challenges because it grants judges broad discretion to detain "
        "defendants and creates a presumption of detention, which may violate the due process rights "
        "of pretrial detainees who are presumed innocent.",
    ),
    (
        "What is e-carceration?",
        "E-carceration refers to the use of electronic monitoring as a condition of pretrial release, "
        "which can be seen as a form of incarceration outside of prison walls, raising concerns that "
        "SB 10 may simply replace physical incarceration with electronic surveillance.",
    ),
    (
        "Are risk assessments used in SB 10 discriminatory?",
        "Risk assessments in SB 10 may be discriminatory because they rely on factors correlated with "
        "race, such as criminal history and socioeconomic status, potentially perpetuating racial bias "
        "in the bail system.",
    ),
]


@pytest.fixture(scope="module")
def setup():
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}. Run: pip install ragas langchain-anthropic datasets")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY not set. Get a free key at console.groq.com")

    from src.utils.embedding_service import EmbeddingService
    from src.utils.es_service import ESService

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es = ESService(es_url)
    try:
        es.es.cluster.health(timeout="2s")
    except Exception:
        pytest.skip("Elasticsearch not available")

    count = es.es.count(
        index="knowledge_base",
        body={"query": {"term": {"file_name": TARGET_FILE}}}
    )
    if count["count"] == 0:
        pytest.skip(f"{TARGET_FILE} not indexed. Upload it via POST /upload first.")

    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    ))

    return es, EmbeddingService(), llm, evaluate, EvaluationDataset, SingleTurnSample, LLMContextPrecisionWithReference, LLMContextRecall


def retrieve_contexts(es_service, embedding_service, question: str, k: int) -> list[str]:
    embedding = embedding_service.encode([question])[0]
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
        fields=["content"],
        knn=body,
    )
    return [hit["fields"]["content"][0] for hit in response["hits"]["hits"]]


def test_ragas_context_metrics(setup):
    es_service, embedding_service, llm, evaluate, EvaluationDataset, SingleTurnSample, LLMContextPrecisionWithReference, LLMContextRecall = setup

    samples = []
    for question, ground_truth in EVAL_SET:
        contexts = retrieve_contexts(es_service, embedding_service, question, TOP_K)
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithReference(llm=llm),
            LLMContextRecall(llm=llm),
        ],
    )

    df = result.to_pandas()
    print("\n  Available columns:", df.columns.tolist())
    print()

    precision_col = [c for c in df.columns if "precision" in c][0]
    recall_col = [c for c in df.columns if "recall" in c][0]

    for _, row in df.iterrows():
        print(
            f"  precision={row[precision_col]:.2f} "
            f"recall={row[recall_col]:.2f}  |  {row['user_input'][:70]}"
        )

    mean_precision = df[precision_col].mean()
    mean_recall = df[recall_col].mean()
    print(f"\n  Mean context precision: {mean_precision:.2f}")
    print(f"  Mean context recall:    {mean_recall:.2f}")

    assert mean_precision >= 0.5, f"Context precision {mean_precision:.2f} below 0.5"
    assert mean_recall >= 0.5, f"Context recall {mean_recall:.2f} below 0.5"
