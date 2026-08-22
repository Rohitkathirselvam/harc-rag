from harc_rag.verification.verifier import LocalVerifier


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


def test_local_verifier_rewrites_unsupported_answer():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        "TCP uses a three-way handshake."
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
        "TCP uses a three-way handshake."
    )

    assert result.is_verified is False

    assert result.confidence == 0.5


def test_local_verifier_keeps_supported_answer():

    verifier = LocalVerifier()

    verifier.llm = FakeLLM(
        "TCP uses a three-way handshake."
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