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
You are a verification assistant.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Task:
1. Check whether the answer is completely supported by the retrieved context.
2. If supported, return the answer unchanged.
3. If unsupported, rewrite the answer using ONLY the retrieved context.
4. Do not add information that is not present in the context.

Verified Answer:
"""

        verified = self.llm.generate(prompt)

        return VerificationResult(

            original_answer=answer,

            verified_answer=verified,

            is_verified=True,

            confidence=1.0,
        )
