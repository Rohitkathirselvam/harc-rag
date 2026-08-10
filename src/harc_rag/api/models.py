from pydantic import BaseModel


class ChatRequest(BaseModel):

    question: str


class SourceChunk(BaseModel):

    source: str | None = None
    text: str
    score: float | None = None


class ChatResponse(BaseModel):

    answer: str
    confidence: float | None = None
    retrieval_confidence: float | None = None
    generation_confidence: float | None = None
    evidence_confidence: float | None = None
    verified: bool | None = None
    verification_reason: str | None = None
    retrieved_chunks: int | None = None
    sources: list[SourceChunk] = []
