from harc_rag.llm.ollama_client import OllamaClient
from harc_rag.verification.models import VerificationResult
from harc_rag.verification.refusal import (
    SAFE_REFUSAL,
    is_safe_refusal,
)


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
You are a strict RAG answer verification system.

Your job is to determine whether the generated answer
is completely supported by the retrieved context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Follow these rules carefully:

1. Check every important factual claim in the generated answer.
2. A claim is SUPPORTED only if the retrieved context contains
   enough information to support it.
3. Do NOT use your general knowledge.
4. Split the question into each requested claim or sub-question.
5. For every claim, decide only SUPPORTED or UNSUPPORTED based
   on the retrieved context. Do not use outside knowledge.
6. Return one block for each claim using exactly:

CLAIM: <requested claim>
STATUS: SUPPORTED or UNSUPPORTED
REASON: <brief evidence-based reason>

7. After all claim blocks, return:

ANSWER: <answer using only supported claims>

8. Include supported claims in ANSWER. If any claim is unsupported,
   explicitly state that the retrieved context does not provide it.
9. An incomplete but evidence-supported answer must retain its
   supported claims; Python will classify it as CORRECTED.
10. If no claim is supported, ANSWER must be the safe refusal:

I don't have enough information from the provided documents
to answer this question.

11. Do not invent facts, including missing numbers, countries,
    percentages, relationships, or causal claims.
12. Do not add explanations outside the retrieved context.

Return EXACTLY this format:

VERDICT: SUPPORTED
REASON: <short explanation>
ANSWER: <final answer>

Python, not the model, determines the aggregate verdict.
"""

        raw = self.llm.generate(prompt).strip()
        reason = "The answer could not be verified against the retrieved context."
        claim_results = []
        current_claim = None
        current_status = None
        current_reason = ""
        answer_lines = []
        answer_started = False

        for line in raw.splitlines():
            stripped = line.strip()

            if stripped.startswith("CLAIM:"):
                if current_claim is not None:
                    claim_results.append({
                        "claim": current_claim,
                        "status": current_status or "UNSUPPORTED",
                        "reason": current_reason,
                    })
                current_claim = stripped.split(":", 1)[1].strip()
                current_status = None
                current_reason = ""
            elif stripped.startswith("STATUS:") and current_claim is not None:
                candidate = stripped.split(":", 1)[1].strip().upper()
                if candidate in {"SUPPORTED", "UNSUPPORTED"}:
                    current_status = candidate
            elif stripped.startswith("REASON:"):
                current_reason = stripped.split(":", 1)[1].strip()
                reason = current_reason
            elif stripped.startswith("ANSWER:"):
                answer_started = True
                answer_lines.append(stripped.split(":", 1)[1].strip())
            elif answer_started:
                answer_lines.append(stripped)

        if current_claim is not None:
            claim_results.append({
                "claim": current_claim,
                "status": current_status or "UNSUPPORTED",
                "reason": current_reason,
            })

        parsed_answer = "\n".join(line for line in answer_lines if line).strip()

        generated_answer_is_refusal = is_safe_refusal(answer)
        parsed_answer_is_refusal = is_safe_refusal(parsed_answer)
        supported_claims = [
            claim
            for claim in claim_results
            if claim["status"] == "SUPPORTED"
        ]
        all_claims_supported = bool(claim_results) and len(supported_claims) == len(claim_results)

        if all_claims_supported and not generated_answer_is_refusal:
            verdict = "SUPPORTED"
            verified_answer = answer.strip()
            is_verified = True
            confidence = 1.0
        elif (
            supported_claims
            and parsed_answer
            and not parsed_answer_is_refusal
        ):
            verdict = "CORRECTED"
            verified_answer = parsed_answer
            is_verified = False
            confidence = 0.75
        else:
            verdict = "UNSUPPORTED"
            verified_answer = SAFE_REFUSAL
            is_verified = False
            confidence = 0.25
            verdict = "UNSUPPORTED"

        return VerificationResult(
            original_answer=answer,
            verified_answer=verified_answer,
            is_verified=is_verified,
            confidence=confidence,
            reason=reason,
            verdict=verdict,
            claim_results=claim_results,
        )

