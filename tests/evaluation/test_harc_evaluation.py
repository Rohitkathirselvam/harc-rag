import json
from pathlib import Path

from harc_rag.chunking.models import Chunk
from harc_rag.pipeline.pipeline import HARCRAGPipeline
from harc_rag.retrieval.models import RetrievalResult

from run_evaluation import calculate_metrics, evaluate_cases


CASES_PATH = Path(__file__).with_name("harc_test_cases.json")


class EvaluationRetriever:

    def retrieve(self, query, k=5):
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=1,
                    text="Development Humanities extends Development Studies through novels, films, poetry, art and music.",
                    start_index=0,
                    end_index=100,
                    metadata={"source": "evaluation"},
                ),
                score=0.9,
            )
        ]


class EvaluationLLM:

    def generate(self, prompt):
        return "Development Humanities extends Development Studies through novels, films, poetry, art and music."


def test_evaluation_cases_are_populated():
    with CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    assert len(cases) >= 20
    assert {case["category"] for case in cases} == {
        "supported",
        "partially_supported",
        "unsupported",
    }
    assert [case["category"] for case in cases].count("supported") >= 10
    assert [case["category"] for case in cases].count("partially_supported") >= 5
    assert [case["category"] for case in cases].count("unsupported") >= 5


def test_evaluation_executes_real_pipeline():
    pipeline = HARCRAGPipeline(
        EvaluationRetriever(),
        llm=EvaluationLLM(),
    )

    records = evaluate_cases(
        pipeline,
        [{
            "question": "Which source materials does Development Humanities use?",
            "category": "supported",
            "expected": "SUPPORTED",
        }],
    )

    assert len(records) == 1
    assert records[0]["final_answer"]
    assert isinstance(records[0]["confidence"], float)
    assert isinstance(records[0]["threshold"], float)
    assert set(calculate_metrics(records)) == {
        "total_questions",
        "verification_rate",
        "unsupported_detection_rate",
        "correction_rate",
        "safe_refusal_rate",
    }
