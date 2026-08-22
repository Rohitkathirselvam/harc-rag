from harc_rag.llm.ollama_client import OllamaClient
from harc_rag.verification.models import VerificationResult


class LocalVerifier:

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
    ):
        self.llm = OllamaClient(
            model=model,
            host=host,
        )

    def verify(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> VerificationResult:

        prompt = f"""
You are a strict answer verification system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Rules:
1. Check whether every important claim in the generated answer is supported by the retrieved context.
2. If the answer is fully supported, return it unchanged.
3. If any claim is unsupported, rewrite the answer using ONLY information from the retrieved context.
4. Never add outside knowledge.
5. Return ONLY the final verified answer.
"""

        verified = self.llm.generate(prompt).strip()

        is_verified = verified == answer.strip()

        confidence = 1.0 if is_verified else 0.5

        return VerificationResult(
            original_answer=answer,
            verified_answer=verified,
            is_verified=is_verified,
            confidence=confidence,
        )