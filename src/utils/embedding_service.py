from sentence_transformers import SentenceTransformer

# BGE models require this prefix on queries (not on indexed documents) for retrieval tasks.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]

    def encode_query(self, text: str) -> list[float]:
        return self.model.encode([_QUERY_PREFIX + text]).tolist()[0]
