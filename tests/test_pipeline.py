from harc_rag.pipeline.pipeline import HARCRAGPipeline
from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult
from harc_rag.verification.models import VerificationResult
from harc_rag.verification.refusal import SAFE_REFUSAL


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


class ModeFakeLLM:

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)

        if "using your general knowledge" in prompt:
            return "Paris is the capital of France."

        return "I don't have enough information from the provided documents."


class EmptyRetriever:

    def retrieve(self, query, k=5):
        return []


class SafeRefusalVerifier:

    def verify(self, question, answer, context):
        return VerificationResult(
            original_answer=answer,
            verified_answer=SAFE_REFUSAL,
            is_verified=False,
            confidence=0.25,
            verdict="UNSUPPORTED",
        )


def test_pipeline_routes_clearly_general_question_to_general_knowledge():

    llm = ModeFakeLLM()
    pipeline = HARCRAGPipeline(FakeRetriever(), llm=llm)
    pipeline.verifier = SafeRefusalVerifier()

    result = pipeline.answer_with_metadata("What is the capital of France?")

    assert result.answer_mode.value == "GENERAL"
    assert result.answer == "Paris is the capital of France."
    assert len(llm.prompts) == 1
    assert "using your general knowledge" in llm.prompts[0]
    assert pipeline._select_answer_mode("What is Python?").value == "GENERAL"
    assert (
        pipeline._select_answer_mode(
            "What is the difference between TCP and UDP?"
        ).value
        == "GENERAL"
    )


def test_pipeline_routes_document_question_to_document_mode():

    llm = ModeFakeLLM()
    pipeline = HARCRAGPipeline(FakeRetriever(), llm=llm)
    pipeline.verifier = SafeRefusalVerifier()

    result = pipeline.answer_with_metadata(
        "What does the uploaded paper say about community arts?"
    )

    assert result.answer_mode.value == "DOCUMENT"
    assert "Answer ONLY using the provided context." in llm.prompts[0]


def test_pipeline_routes_ambiguous_question_to_document_mode():

    llm = ModeFakeLLM()
    pipeline = HARCRAGPipeline(FakeRetriever(), llm=llm)
    pipeline.verifier = SafeRefusalVerifier()

    result = pipeline.answer_with_metadata("What is it?")

    assert result.answer_mode.value == "DOCUMENT"
    assert "Answer ONLY using the provided context." in llm.prompts[0]


def test_document_question_with_insufficient_context_stays_document_mode():

    llm = ModeFakeLLM()
    pipeline = HARCRAGPipeline(EmptyRetriever(), llm=llm)
    pipeline.verifier = SafeRefusalVerifier()

    def general_prompt_must_not_be_called(*args, **kwargs):
        raise AssertionError("DOCUMENT questions must not use the general prompt")

    pipeline.prompt_builder.build_general = general_prompt_must_not_be_called

    result = pipeline.answer_with_metadata(
        "According to the uploaded paper, what was Japan's GDP in 2026?"
    )

    assert result.answer_mode.value == "DOCUMENT"
    assert result.answer == SAFE_REFUSAL
    assert result.verification_verdict == "UNSUPPORTED"
    assert len(llm.prompts) == 1
    assert "Answer ONLY using the provided context." in llm.prompts[0]


def test_general_answers_do_not_include_document_metadata():

    pipeline = HARCRAGPipeline(FakeRetriever(), llm=ModeFakeLLM())

    result = pipeline.answer_with_metadata("Explain what an API is.")

    assert result.answer_mode.value == "GENERAL"
    assert result.sources == []
    assert result.retrieved_chunks == 0
    assert result.confidence is None
    assert result.retrieval_confidence is None
    assert result.generation_confidence is None
    assert result.evidence_confidence is None
    assert result.verified is None
    assert result.verification_reason is None
    assert result.verification_verdict is None


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
