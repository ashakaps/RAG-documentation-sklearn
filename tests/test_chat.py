"""Тесты API без внешних зависимостей: ни эмбеддера, ни Qdrant, ни LLM."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_rag():
    """Подменяем build_rag_chain для КАЖДОГО теста файла.

    lifespan вызывает build_rag_chain при старте TestClient; без подмены
    это грузит эмбеддер (~118 МБ) и ходит в Qdrant.
    """
    chain = MagicMock()
    chain.invoke.return_value = "Ridge uses L2 penalty [1]."

    doc = MagicMock()
    doc.metadata = {"source": "https://scikit-learn.org/stable/linear.html"}
    doc.page_content = "Ridge regression addresses..."
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]

    with patch("app.main.build_rag_chain", return_value=(chain, retriever)):
        yield chain, retriever


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_validates_empty_question() -> None:
    """Пустой вопрос отсекает Pydantic ещё до цепочки."""
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_returns_answer_with_sources() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 200
    body = response.json()
    assert "Ridge uses L2" in body["answer"]
    assert len(body["sources"]) == 1
    assert "scikit-learn.org" in body["sources"][0]["url"]


def test_chat_returns_503_when_llm_fails(mock_rag) -> None:
    """Падение LLM-провайдера (403/402/429) превращается в 503, а не в 500."""
    chain, _ = mock_rag
    chain.invoke.side_effect = RuntimeError("provider down")

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 503
    assert "LLM provider" in response.json()["detail"]
