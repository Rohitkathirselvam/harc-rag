from harc_rag.routing.router import AdaptiveRouter


def test_router_skips_verification_when_confidence_is_high():

    router = AdaptiveRouter()

    decision = router.route(
        confidence=0.90,
        question="What is TCP?",
        answer="TCP is a transport protocol.",
        context="TCP is a transport protocol.",
    )

    assert decision.should_verify is False
    assert decision.confidence == 0.90
    assert "Sufficient confidence" in decision.reason


def test_router_requests_verification_when_confidence_is_low():

    router = AdaptiveRouter()

    decision = router.route(
        confidence=0.40,
        question="What is TCP?",
        answer="TCP uses a three-way handshake.",
        context="TCP uses a three-way handshake.",
    )

    assert decision.should_verify is True
    assert decision.confidence == 0.40
    assert "verification" in decision.reason.lower()


def test_router_uses_lower_threshold_for_complex_questions():

    router = AdaptiveRouter()

    decision = router.route(
        confidence=0.65,
        question=(
            "What are the main differences between TCP and UDP "
            "in terms of reliability and connection establishment?"
        ),
        answer="TCP is connection-oriented.",
        context="TCP is connection-oriented.",
    )

    # Complex questions use threshold 0.60.
    # 0.65 is therefore considered sufficient.
    assert decision.should_verify is False