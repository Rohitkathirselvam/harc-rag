from dataclasses import dataclass
from os import getenv
from typing import Any

from harc_rag.generation.generator import RAGGenerator
from harc_rag.generation.interfaces import LLM
from harc_rag.generation.ollama import OllamaLLM
from harc_rag.generation.service import GenerationService
from harc_rag.generation.prompt_builder import PromptBuilder

from harc_rag.routing.router import AdaptiveRouter
from harc_rag.uncertainty.estimator import JointEstimator
from harc_rag.verification.verifier import LocalVerifier


# ============================================================
# PIPELINE SOURCE
# ============================================================

@dataclass
class PipelineSource:
    source: str | None
    text: str
    score: float | None


# ============================================================
# PIPELINE ANSWER
# ============================================================

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


# ============================================================
# HALLUCINATION-AWARE RAG PIPELINE
# ============================================================

class HARCRAGPipeline:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        retriever,
        llm: LLM | None = None,
    ):
        self.retriever = retriever

        # Prompt builder
        self.prompt_builder = PromptBuilder()

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        default_llm = llm or OllamaLLM(
            model=getenv(
                "HARC_RAG_MODEL",
                "qwen2.5:3b",
            ),
            host=getenv(
                "HARC_RAG_OLLAMA_HOST",
                "http://127.0.0.1:11434",
            ),
        )

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        self.generator = RAGGenerator(
            GenerationService(default_llm)
        )

        # ----------------------------------------------------
        # Uncertainty estimator
        # ----------------------------------------------------

        self.estimator = JointEstimator()

        # ----------------------------------------------------
        # Adaptive router
        # ----------------------------------------------------

        self.router = AdaptiveRouter()

        # ----------------------------------------------------
        # Local verifier
        # ----------------------------------------------------

        self.verifier = LocalVerifier(
            model=getenv(
                "HARC_RAG_MODEL",
                "qwen2.5:3b",
            ),
            host=getenv(
                "HARC_RAG_OLLAMA_HOST",
                "http://127.0.0.1:11434",
            ),
        )

    # ========================================================
    # SIMPLE ANSWER API
    # ========================================================

    def answer(
        self,
        question: str,
    ) -> str:

        result = self.answer_with_metadata(question)

        return result.answer

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def answer_with_metadata(
        self,
        question: str,
    ) -> PipelineAnswer:

        # 1. Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(
            question
        )

        chunks = [
            result.chunk
            for result in retrieval_results
        ]

        # 2. Build context
        context = "\n".join(
            chunk.text
            for chunk in chunks
        )

        # 3. Build RAG prompt
        prompt = self.prompt_builder.build(
            query=question,
            chunks=chunks,
        )

        # 4. Generate RAG answer
        answer = self.generator.generate(
            prompt
        )

        # 5. Check whether fallback is required
        used_general_fallback = (
            not retrieval_results
            or not context.strip()
            or self._needs_general_fallback(answer)
        )

        # 6. General LLM fallback
        if used_general_fallback:

            fallback_prompt = (
                self.prompt_builder.build_general(
                    query=question
                )
            )

            # Replace the original RAG answer
            # with the general LLM answer.
            answer = self.generator.generate(
                fallback_prompt
            )

        # 7. Estimate uncertainty
        uncertainty = self.estimator.estimate(
            retrieval_results,
            answer,
            context,
        )

        # 8. Adaptive routing
        decision = self.router.route(
            confidence=uncertainty.score,
            question=question,
            answer=answer,
            context=context,
        )

        # 9. Default values
        final_answer = answer
        verified = False
        verification_reason = decision.reason

        # 10. If fallback was used,
        # don't run the local verifier.
        if used_general_fallback:

            final_answer = answer

            verified = False

            verification_reason = (
                "The uploaded documents did not contain enough "
                "information, so the answer was generated from "
                "the LLM's general knowledge."
            )

        # 11. Normal RAG verification
        elif decision.should_verify:

            # Use a smaller evidence context for verification.
            # This prevents the verifier from receiving an unnecessarily
            # large retrieved context.
            verification_context = "\n\n".join(
                result.chunk.text
                for result in retrieval_results[:3]
            )

            verification = self.verifier.verify(
                question,
                answer,
                verification_context,
            )

            final_answer = verification.verified_answer
            verified = verification.is_verified

            verification_reason = getattr(
                verification,
                "reason",
                verification_reason,
            )

        # 12. Build sources
        sources = self._build_sources(
            retrieval_results
        )

        # 13. Return result
        return PipelineAnswer(
            answer=final_answer,
            confidence=uncertainty.score,
            retrieval_confidence=(
                uncertainty.confidence.retrieval
            ),
            generation_confidence=(
                uncertainty.confidence.generation
            ),
            evidence_confidence=(
                uncertainty.confidence.evidence
            ),
            verified=verified,
            verification_reason=verification_reason,
            retrieved_chunks=len(
                retrieval_results
            ),
            sources=sources,
        )

    def _build_sources(
        self,
        retrieval_results,
    ) -> list[PipelineSource]:
        """
        Convert retrieval results into PipelineSource objects.
        """

        sources: list[PipelineSource] = []

        for result in retrieval_results:

            chunk = result.chunk

            # Safely handle missing metadata
            metadata = (
                chunk.metadata
                if getattr(chunk, "metadata", None)
                else {}
            )

            source = metadata.get(
                "source"
            )

            text = getattr(
                chunk,
                "text",
                "",
            )

            score = getattr(
                result,
                "score",
                None,
            )

            sources.append(
                PipelineSource(
                    source=source,
                    text=text,
                    score=score,
                )
            )

        return sources

    # ========================================================
    # GENERAL LLM FALLBACK DETECTION
    # ========================================================

    def _needs_general_fallback(
        self,
        answer: str,
    ) -> bool:
        """
        Detect whether the RAG-generated answer indicates
        that the retrieved context is insufficient.

        Returns True when a second LLM call should be made
        using the general-knowledge prompt.
        """

        # Empty answer
        if not answer:

            return True

        answer_lower = (
            str(answer)
            .strip()
            .lower()
        )

        # ----------------------------------------------------
        # Common insufficient-context phrases
        # ----------------------------------------------------

        insufficient_phrases = (

            "i don't have enough information",

            "i do not have enough information",

            "not enough information",

            "insufficient information",

            "i need more information",

            "i need additional information",

            "cannot determine",

            "can't determine",

            "cannot be determined",

            "can't be determined",

            "cannot answer from the provided context",

            "cannot answer from the provided documents",

            "cannot answer using the provided context",

            "cannot answer using the provided documents",

            "not found in the provided context",

            "not found in the provided documents",

            "not found in the retrieved context",

            "does not contain enough information",

            "does not contain sufficient information",

            "do not contain enough information",

            "do not contain sufficient information",

            "provided context does not contain",

            "provided documents do not contain",

            "uploaded documents did not contain",

            "could not find that information",

            "couldn't find that information",

            "generated answer is not supported by retrieved context",

            "generated answer is not supported by the retrieved context",

            "generated answer is not supported by provided context",

            "generated answer is not supported by the provided context",

            "generated answer is not supported by provided documents",

            "generated answer is not supported by the provided documents",

            "revised answer using only information from the retrieved context",

            "revised answer using only information from the provided context",

            "revised answer using only information from the retrieved documents",

            "revised answer using only information from the provided documents",
        )

        # ----------------------------------------------------
        # Direct phrase matching
        # ----------------------------------------------------

        if any(
            phrase in answer_lower
            for phrase in insufficient_phrases
        ):

            return True

        # ----------------------------------------------------
        # Pattern 1:
        #
        # "The generated answer is not supported
        #  by the retrieved context."
        # ----------------------------------------------------

        unsupported_answer = (
            "generated answer" in answer_lower
            and "not supported" in answer_lower
            and (
                "retrieved context"
                in answer_lower
                or
                "provided context"
                in answer_lower
                or
                "provided documents"
                in answer_lower
                or
                "retrieved documents"
                in answer_lower
            )
        )

        if unsupported_answer:

            return True

        # ----------------------------------------------------
        # Pattern 2:
        #
        # "A revised answer cannot be provided."
        # ----------------------------------------------------

        revised_answer_unavailable = (
            "revised answer" in answer_lower
            and (
                "cannot be provided"
                in answer_lower
                or
                "can't be provided"
                in answer_lower
                or
                "could not be provided"
                in answer_lower
                or
                "couldn't be provided"
                in answer_lower
            )
        )

        if revised_answer_unavailable:

            return True

        # ----------------------------------------------------
        # Pattern 3:
        #
        # Answer explicitly says context is insufficient
        # ----------------------------------------------------

        context_insufficient = (
            (
                "context is insufficient"
                in answer_lower
            )
            or
            (
                "context is not sufficient"
                in answer_lower
            )
            or
            (
                "context does not provide enough"
                in answer_lower
            )
            or
            (
                "context does not provide sufficient"
                in answer_lower
            )
        )

        if context_insufficient:

            return True

        # ----------------------------------------------------
        # Pattern 4:
        #
        # Answer says information cannot be found in
        # uploaded documents.
        # ----------------------------------------------------

        document_information_missing = (
            (
                "information"
                in answer_lower
            )
            and (
                "not found"
                in answer_lower
                or
                "cannot be found"
                in answer_lower
                or
                "could not be found"
                in answer_lower
            )
            and (
                "document"
                in answer_lower
                or
                "context"
                in answer_lower
            )
        )

        if document_information_missing:

            return True

        # ----------------------------------------------------
        # Otherwise RAG answer is considered sufficient
        # ----------------------------------------------------

        return False