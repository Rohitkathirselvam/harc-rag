from harc_rag.pipeline.pipeline import HARCRAGPipeline
from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult
from harc_rag.verification.models import VerificationResult


class FakeRetriever:

    def retrieve(
        self,
        query,
        k=5,
    ):

        return [

            RetrievalResult(

                chunk=Chunk(
                    chunk_id=1,
                    text="TCP uses a three-way handshake.",
                    start_index=0,
                    end_index=30,
                    metadata={},
                ),

                score=0.9,
            )

        ]


def test_pipeline_creation():

    pipeline = HARCRAGPipeline(
        FakeRetriever()
    )

    assert pipeline is not None


class FakeLLM:

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)

        if "Answer ONLY using the provided context." in prompt:
            return "I don't have enough information from the provided documents."

        return "LLM stands for large language model."


def test_pipeline_keeps_rag_answer_when_context_is_insufficient():

    llm = FakeLLM()

    pipeline = HARCRAGPipeline(
        FakeRetriever(),
        llm=llm,
    )

    result = pipeline.answer_with_metadata("What is LLM?")

    assert result.answer == (
        "I don't have enough information from the provided "
        "documents to answer this question."
    )
    assert result.original_answer == (
        "I don't have enough information from the provided documents."
    )
    assert result.verification_verdict == "UNSUPPORTED"
    assert len(llm.prompts) == 1
    assert "general knowledge" not in result.verification_reason


class LowConfidenceRetriever:

    def retrieve(
        self,
        query,
        k=5,
    ):

        return [

            RetrievalResult(

                chunk=Chunk(
                    chunk_id=1,
                    text="TCP uses a three-way handshake.",
                    start_index=0,
                    end_index=40,
                    metadata={},
                ),

                score=0.1,
            )

        ]


class VerificationFakeVerifier:

    def __init__(self):
        self.called = False

    def verify(
        self,
        question,
        answer,
        context,
    ):

        self.called = True

        return VerificationResult(
            original_answer=answer,
            verified_answer="TCP uses a three-way handshake.",
            is_verified=False,
            confidence=0.5,
        )


class VerificationLLM:

    def generate(self, prompt):

        return "TCP uses a four-way handshake."


def test_pipeline_verifies_low_confidence_answer():

    pipeline = HARCRAGPipeline(
        LowConfidenceRetriever(),
        llm=VerificationLLM(),
    )

    verifier = VerificationFakeVerifier()

    pipeline.verifier = verifier

    result = pipeline.answer_with_metadata(
        "How does TCP connect?"
    )

    assert verifier.called is True

    assert result.answer == (
        "TCP uses a three-way handshake."
    )

    assert result.verified is False
