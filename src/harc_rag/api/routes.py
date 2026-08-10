from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from harc_rag.api.models import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
)


router = APIRouter()


_pipeline = None

_indexer = None


UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def set_pipeline(pipeline):

    global _pipeline

    _pipeline = pipeline


def set_indexer(indexer):

    global _indexer

    _indexer = indexer


@router.get("/health")
def health():

    return {
        "status": "ok"
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    if _pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="HARC-RAG pipeline is not initialized",
        )

    if not hasattr(_pipeline, "answer_with_metadata"):
        return ChatResponse(
            answer=_pipeline.answer(request.question)
        )

    result = _pipeline.answer_with_metadata(request.question)

    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        retrieval_confidence=result.retrieval_confidence,
        generation_confidence=result.generation_confidence,
        evidence_confidence=result.evidence_confidence,
        verified=result.verified,
        verification_reason=result.verification_reason,
        retrieved_chunks=result.retrieved_chunks,
        sources=[
            SourceChunk(
                source=source.source,
                text=source.text,
                score=source.score,
            )
            for source in result.sources
        ],
    )


@router.post(
    "/documents/upload"
)
async def upload_document(
    file: UploadFile = File(...),
):

    # Check filename
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    # Only allow PDF
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    # Save uploaded file
    file_path = (
        UPLOAD_DIR /
        file.filename
    )

    contents = await file.read()

    file_path.write_bytes(
        contents
    )

    # Index document
    indexed_chunks = None

    if _indexer is not None:

        indexed_chunks = _indexer.index(
            file_path
        )

    return {
        "message": (
            "PDF uploaded and indexed successfully"
            if indexed_chunks is not None
            else "PDF uploaded successfully"
        ),
        "filename": file.filename,
        "chunks": (
            len(indexed_chunks)
            if indexed_chunks is not None
            else 0
        ),
    }
