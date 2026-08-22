from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult
from harc_rag.uncertainty.estimator import JointEstimator


def create_retrieval_results():
    return [
        RetrievalResult(
            chunk=Chunk(
                chunk_id=1,
                text="TCP uses a three-way handshake.",
                start_index=0,
                end_index=35,
                metadata={},
            ),
            score=0.9,
        ),
        RetrievalResult(
            chunk=Chunk(
                chunk_id=2,
                text="TCP provides reliable communication.",
                start_index=36,
                end_index=75,
                metadata={},
            ),
            score=0.8,
        ),
    ]


def test_joint_estimator_returns_confidence():

    estimator = JointEstimator()

    # Avoid loading the SentenceTransformer model during
    # this unit test. Evidence confidence is tested separately.
    estimator.evidence.estimate = (
        lambda answer, context: 0.8
    )

    result = estimator.estimate(
        retrieval_results=create_retrieval_results(),
        answer="TCP uses a three-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    assert result is not None
    assert result.confidence is not None

    assert 0.0 <= result.confidence.retrieval <= 1.0
    assert 0.0 <= result.confidence.generation <= 1.0
    assert 0.0 <= result.confidence.evidence <= 1.0

    assert 0.0 <= result.score <= 1.0


def test_joint_estimator_combines_confidence_scores():

    estimator = JointEstimator()

    # Make the three confidence components deterministic.
    estimator.retrieval.estimate = (
        lambda results: 0.8
    )

    estimator.generation.estimate = (
        lambda answer: 0.9
    )

    estimator.evidence.estimate = (
        lambda answer, context: 0.7
    )

    result = estimator.estimate(
        retrieval_results=create_retrieval_results(),
        answer="TCP uses a three-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    # Retrieval confidence = 0.8
    #
    # DynamicWeightCalculator should therefore use:
    # retrieval = 0.50
    # generation = 0.20
    # evidence = 0.30
    #
    # Expected:
    #
    # (0.50 * 0.8)
    # + (0.20 * 0.9)
    # + (0.30 * 0.7)
    #
    # = 0.40 + 0.18 + 0.21
    # = 0.79

    expected = 0.79

    assert result.score == expected


def test_joint_estimator_does_not_decide_verification():

    estimator = JointEstimator()

    estimator.retrieval.estimate = (
        lambda results: 0.2
    )

    estimator.generation.estimate = (
        lambda answer: 0.3
    )

    estimator.evidence.estimate = (
        lambda answer, context: 0.1
    )

    result = estimator.estimate(
        retrieval_results=create_retrieval_results(),
        answer="Uncertain answer.",
        context="Some context.",
    )

    # JointEstimator calculates confidence.
    # AdaptiveRouter is responsible for deciding
    # whether verification should happen.

    assert result.should_verify is False