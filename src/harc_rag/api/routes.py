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
from harc_rag.memory.manager import MemoryManager
from harc_rag.memory.context_builder import ContextBuilder


router = APIRouter()


_pipeline = None

_indexer = None

_memory_manager = MemoryManager()
_context_builder = ContextBuilder()


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

    # ---------------------------------------
    # CREATE / LOAD CONVERSATION
    # ---------------------------------------

    conversation_id = request.conversation_id

    if not conversation_id:

        import uuid

        conversation_id = str(uuid.uuid4())

        conversation = _memory_manager.create(
            conversation_id
        )

    else:

        conversation = _memory_manager.get(
            conversation_id
        )

    # ---------------------------------------
    # BUILD CONVERSATION CONTEXT
    # ---------------------------------------

    conversation_context = _context_builder.build(
        conversation
    )

    # ---------------------------------------
    # SAVE USER MESSAGE
    # ---------------------------------------

    _memory_manager.add_message(
        conversation_id,
        "user",
        request.question,
    )

    # ---------------------------------------
    # RUN HARC-RAG
    # ---------------------------------------

    if hasattr(
        _pipeline,
        "answer_with_metadata"
    ):

        result = _pipeline.answer_with_metadata(
            request.question,
            conversation_context=conversation_context,
        )

        answer = result.answer

    else:

        answer = _pipeline.answer(
            request.question
        )

        result = None

    # ---------------------------------------
    # SAVE ASSISTANT MESSAGE
    # ---------------------------------------

    _memory_manager.add_message(
        conversation_id,
        "assistant",
        answer,
    )

    # ---------------------------------------
    # RESPONSE
    # ---------------------------------------

    if result is None:

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
        )

    return ChatResponse(
        answer=result.answer,
        original_answer=result.original_answer,
        conversation_id=conversation_id,
        confidence=result.confidence,
        retrieval_confidence=result.retrieval_confidence,
        generation_confidence=result.generation_confidence,
        evidence_confidence=result.evidence_confidence,
        verified=result.verified,
        verification_reason=result.verification_reason,
        verification_verdict=result.verification_verdict,
        retrieved_chunks=result.retrieved_chunks,
        answer_mode=result.answer_mode.value,
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
