from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from harc_rag.api.indexing import DocumentIndexingService
from harc_rag.api.routes import router, set_indexer, set_pipeline
from harc_rag.chunking.character_strategy import CharacterChunkingStrategy
from harc_rag.chunking.splitter import TextSplitter
from harc_rag.document.loader import DocumentLoader
from harc_rag.embedding.minilm import MiniLMEmbeddingModel
from harc_rag.embedding.service import EmbeddingService
from harc_rag.pipeline.pipeline import HARCRAGPipeline
from harc_rag.retrieval.bm25_retriever import BM25Retriever
from harc_rag.retrieval.dense_retriever import DenseRetriever
from harc_rag.retrieval.hybrid_retriever import HybridRetriever
from harc_rag.vectorstore.faiss_store import FAISSVectorStore


app = FastAPI(
    title="HARC-RAG API",
    description="Hallucination-Aware Retrieval-Augmented Generation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def initialize_harc_rag():

    embedding_model = MiniLMEmbeddingModel()
    embedding_service = EmbeddingService(embedding_model)
    vector_store = FAISSVectorStore()
    bm25_retriever = BM25Retriever([])
    dense_retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    indexer = DocumentIndexingService(
        loader=DocumentLoader(),
        splitter=TextSplitter(CharacterChunkingStrategy()),
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    set_indexer(indexer)
    set_pipeline(HARCRAGPipeline(retriever))


app.include_router(router)

frontend_dir = Path(__file__).resolve().parents[3] / "frontend"

if frontend_dir.exists():
    app.mount(
        "/app",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )


@app.get("/")
def root():

    return {
        "message": "HARC-RAG API is running"
    }
