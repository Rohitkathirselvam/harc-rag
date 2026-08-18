from harc_rag.pipeline.pipeline import HARCRAGPipeline
from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult


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


def test_pipeline_falls_back_to_general_llm_when_context_is_insufficient():

    llm = FakeLLM()

    pipeline = HARCRAGPipeline(
        FakeRetriever(),
        llm=llm,
    )

    result = pipeline.answer_with_metadata("What is LLM?")

    assert result.answer == "LLM stands for large language model."
    assert len(llm.prompts) == 2
    assert "general knowledge" in llm.prompts[1]
    assert "general knowledge" in result.verification_reason
