import os
import pytest
from unittest.mock import Mock

from src.utils.embedding_service import EmbeddingService


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires running Elasticsearch")


@pytest.fixture
def mock_embedding_service():
    svc = Mock(spec=EmbeddingService)
    svc.encode.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    svc.encode_one.return_value = [0.1] * 1024
    svc.encode_query.return_value = [0.1] * 1024
    return svc


@pytest.fixture(scope="session")
def es_service():
    from src.utils.es_service import ESService
    svc = ESService(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    try:
        svc.es.cluster.health(timeout="2s")
    except Exception:
        pytest.skip("Elasticsearch not available")
    return svc
