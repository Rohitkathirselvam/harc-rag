import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harc_rag.api.indexing import DocumentIndexingService
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
from harc_rag.verification.refusal import is_safe_refusal


CASES_PATH = Path(__file__).with_name("harc_test_cases.json")
RESULTS_PATH = Path(__file__).with_name("results.json")
DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "Towards a  Development Humanities   widening the multi-disciplinary field of development studies.pdf"
)


def build_pipeline():
    embedding_model = MiniLMEmbeddingModel()
    embedding_service = EmbeddingService(embedding_model)
    vector_store = FAISSVectorStore()
    bm25_retriever = BM25Retriever([])
    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        ),
        bm25_retriever=bm25_retriever,
    )
    indexer = DocumentIndexingService(
        loader=DocumentLoader(),
        splitter=TextSplitter(CharacterChunkingStrategy()),
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )
    indexer.index(DOCUMENT_PATH)
    return HARCRAGPipeline(retriever)


def verification_status(result, verification_triggered):
    if not verification_triggered:
        return "NOT_REQUIRED"
    return result.verification_verdict


def evaluate_cases(pipeline, cases):
    records = []
    for case in cases:
        result = pipeline.answer_with_metadata(case["question"])
        threshold = pipeline.router.threshold.calculate(case["question"])
        verification_triggered = result.confidence < threshold
        records.append(
            {
                "question": case["question"],
                "category": case["category"],
                "expected": case["expected"],
                "original_answer": result.original_answer,
                "confidence": result.confidence,
                "threshold": threshold,
                "retrieval_confidence": result.retrieval_confidence,
                "generation_confidence": result.generation_confidence,
                "evidence_confidence": result.evidence_confidence,
                "verification_triggered": verification_triggered,
                "verification_status": verification_status(
                    result,
                    verification_triggered,
                ),
                "final_answer": result.answer,
            }
        )
    return records


def calculate_metrics(records):
    total = len(records)
    unsupported = [record for record in records if record["category"] == "unsupported"]
    partial = [record for record in records if record["category"] == "partially_supported"]

    def rate(matches, population):
        return sum(matches) / len(population) if population else 0.0

    return {
        "total_questions": total,
        "verification_rate": sum(record["verification_triggered"] for record in records) / total,
        "unsupported_detection_rate": rate(
            [record["verification_status"] == "UNSUPPORTED" for record in unsupported],
            unsupported,
        ),
        "correction_rate": rate(
            [record["verification_status"] == "CORRECTED" for record in partial],
            partial,
        ),
        "safe_refusal_rate": rate(
            [is_safe_refusal(record["final_answer"]) for record in unsupported],
            unsupported,
        ),
    }


def print_summary(records, metrics):
    headers = ["#", "Category", "Confidence", "Threshold", "Verify", "Status"]
    print(" | ".join(headers))
    print("-+-".join("-" * len(header) for header in headers))
    for index, record in enumerate(records, 1):
        print(
            f"{index:>2} | {record['category']:<19} | "
            f"{record['confidence']:.3f}      | {record['threshold']:.3f}    | "
            f"{str(record['verification_triggered']):<5}  | {record['verification_status']}"
        )
    print("\nMetrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}" if name != "total_questions" else f"{name}: {value}")


def main():
    with CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)
    if len(cases) < 20:
        raise ValueError("The evaluation requires at least 20 test cases.")
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"Development Studies PDF not found: {DOCUMENT_PATH}")

    pipeline = build_pipeline()
    records = evaluate_cases(pipeline, cases)
    metrics = calculate_metrics(records)
    payload = {
        "source_document": DOCUMENT_PATH.name,
        "results": records,
        "metrics": metrics,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
    print_summary(records, metrics)


if __name__ == "__main__":
    main()
