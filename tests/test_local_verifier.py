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


def test_local_verifier_supports_multiple_claims():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        """CLAIM: What protocol is described?
STATUS: SUPPORTED
REASON: The context describes TCP.
CLAIM: What handshake does it use?
STATUS: SUPPORTED
REASON: The context describes a three-way handshake.
ANSWER: TCP uses a three-way handshake."""
    )

    result = verifier.verify(
        question="What protocol is described and what handshake does it use?",
        answer="TCP uses a three-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    assert result.verdict == "SUPPORTED"
    assert len(result.claim_results) == 2


def test_local_verifier_aggregates_partial_claim_status():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        """CLAIM: What source materials are discussed?
STATUS: PARTIALLY_SUPPORTED
REASON: The context lists novels and films.
CLAIM: What percentage uses them?
STATUS: UNSUPPORTED
REASON: No percentage is provided.
ANSWER: The context discusses novels and films. The percentage is not provided."""
    )

    result = verifier.verify(
        question="What source materials are discussed and what percentage uses them?",
        answer="The percentage is unavailable.",
        context="The paper discusses novels and films.",
    )

    assert result.verdict == "CORRECTED"
    assert "novels and films" in result.verified_answer
    assert "percentage is not provided" in result.verified_answer


def test_local_verifier_fails_safely_on_malformed_output():

    verifier = LocalVerifier()
    verifier.llm = FakeLLM("This is not structured verifier output.")

    result = verifier.verify(
        question="What is the GDP of Japan?",
        answer="The GDP is unavailable.",
        context="TCP uses a three-way handshake.",
    )

    assert result.verdict == "UNSUPPORTED"
    assert result.verified_answer == (
        "I don't have enough information from the provided documents "
        "to answer this question."
    )