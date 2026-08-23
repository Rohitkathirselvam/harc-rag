from harc_rag.verification.verifier import LocalVerifier


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


def test_local_verifier_rewrites_unsupported_answer():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        """CLAIM: What handshake does TCP use?
    STATUS: SUPPORTED
    REASON: The context states that TCP uses a three-way handshake.
    CLAIM: What transport does UDP use?
    STATUS: UNSUPPORTED
    REASON: The context does not mention UDP.
    ANSWER: TCP uses a three-way handshake. The context does not provide information about UDP."""
    )

    result = verifier.verify(
        question="How does TCP connect?",
        answer="TCP uses a four-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    assert result.original_answer == (
        "TCP uses a four-way handshake."
    )

    assert result.verified_answer == (
        "TCP uses a three-way handshake. "
        "The context does not provide information about UDP."
    )

    assert result.is_verified is False

    assert result.confidence == 0.75
    assert result.verdict == "CORRECTED"


def test_local_verifier_keeps_supported_answer():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        """CLAIM: What handshake does TCP use?
    STATUS: SUPPORTED
    REASON: The context states that TCP uses a three-way handshake.
    ANSWER: TCP uses a three-way handshake."""
    )

    result = verifier.verify(
        question="How does TCP connect?",
        answer="TCP uses a three-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    assert result.original_answer == (
        "TCP uses a three-way handshake."
    )

    assert result.verified_answer == (
        "TCP uses a three-way handshake."
    )

    assert result.is_verified is True

    assert result.confidence == 1.0
    assert result.verdict == "SUPPORTED"


def test_local_verifier_refuses_when_no_claims_are_supported():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        """CLAIM: What is the GDP of Japan?
STATUS: UNSUPPORTED
REASON: The context contains no GDP information.
ANSWER: I don't have enough information from the provided documents to answer this question."""
    )

    result = verifier.verify(
        question="What is the GDP of Japan?",
        answer="Japan's GDP is unavailable in the provided documents.",
        context="TCP uses a three-way handshake.",
    )

    assert result.verdict == "UNSUPPORTED"
    assert result.verified_answer == (
        "I don't have enough information from the provided documents "
        "to answer this question."
    )
    assert result.is_verified is False