from fastapi.testclient import TestClient

from harc_rag.api.main import app
from harc_rag.api.models import ChatResponse
from harc_rag.api.routes import set_pipeline


class FakePipeline:

    def answer(self, question: str) -> str:

        return f"Answer for: {question}"


def test_chat():

    set_pipeline(FakePipeline())

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "What is TCP?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data

    assert data["answer"] == "Answer for: What is TCP?"
    assert data["answer_mode"] == "DOCUMENT"


def test_chat_response_exposes_general_answer_mode_without_sources():

    response = ChatResponse(
        answer="Paris is the capital of France.",
        answer_mode="GENERAL",
        sources=[],
    )

    assert response.answer_mode == "GENERAL"
    assert response.sources == []
    assert response.confidence is None
    assert response.retrieval_confidence is None
    assert response.evidence_confidence is None
    assert response.verified is None
