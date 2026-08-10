from dataclasses import dataclass
from os import getenv

from harc_rag.generation.generator import RAGGenerator
from harc_rag.generation.interfaces import LLM
from harc_rag.generation.ollama import OllamaLLM
from harc_rag.generation.service import GenerationService
from harc_rag.generation.prompt_builder import PromptBuilder

from harc_rag.routing.router import AdaptiveRouter
from harc_rag.uncertainty.estimator import JointEstimator
from harc_rag.verification.verifier import LocalVerifier


@dataclass
class PipelineSource:

    source: str | None
    text: str
    score: float | None


@dataclass
class PipelineAnswer:

    answer: str
    confidence: float | None
    retrieval_confidence: float | None
    generation_confidence: float | None
    evidence_confidence: float | None
    verified: bool | None
    verification_reason: str | None
    retrieved_chunks: int
    sources: list[PipelineSource]


class HARCRAGPipeline:

    def __init__(
        self,
        retriever,
        llm: LLM | None = None,
    ):

        self.retriever = retriever

        self.prompt_builder = PromptBuilder()

        default_llm = llm or OllamaLLM(
            model=getenv("HARC_RAG_MODEL", "qwen2.5:3b"),
            host=getenv("HARC_RAG_OLLAMA_HOST", "http://127.0.0.1:11434"),
        )

        self.generator = RAGGenerator(GenerationService(default_llm))

        self.estimator = JointEstimator()

        self.router = AdaptiveRouter()

        self.verifier = LocalVerifier(
            model=getenv("HARC_RAG_MODEL", "qwen2.5:3b"),
            host=getenv("HARC_RAG_OLLAMA_HOST", "http://127.0.0.1:11434"),
        )

    def answer(
        self,
        question: str,
    ) -> str:

        return self.answer_with_metadata(question).answer

    def answer_with_metadata(
        self,
        question: str,
    ) -> PipelineAnswer:

        # Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(question)

        chunks = [
            result.chunk
            for result in retrieval_results
        ]

        context = "\n".join(
            chunk.text
            for chunk in chunks
        )

        # Build prompt
        prompt = self.prompt_builder.build(
            query=question,
            chunks=chunks,
        )

        # Generate answer
        answer = self.generator.generate(prompt)

        # Estimate uncertainty
        uncertainty = self.estimator.estimate(
            retrieval_results,
            answer,
            context,
        )

        # Decide whether verification is required
        decision = self.router.route(
            confidence=uncertainty.score,
            question=question,
            answer=answer,
            context=context,
        )

        verified = False
        verification_reason = decision.reason
        final_answer = answer

        if decision.should_verify:

            verification = self.verifier.verify(
                question,
                answer,
                context,
            )

            final_answer = verification.verified_answer
            verified = verification.is_verified

        return PipelineAnswer(
            answer=final_answer,
            confidence=uncertainty.score,
            retrieval_confidence=uncertainty.confidence.retrieval,
            generation_confidence=uncertainty.confidence.generation,
            evidence_confidence=uncertainty.confidence.evidence,
            verified=verified,
            verification_reason=verification_reason,
            retrieved_chunks=len(retrieval_results),
            sources=[
                PipelineSource(
                    source=result.chunk.metadata.get("source"),
                    text=result.chunk.text,
                    score=result.score,
                )
                for result in retrieval_results
            ],
        )
