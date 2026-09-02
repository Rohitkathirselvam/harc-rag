from harc_rag.routing.models import RoutingDecision
from harc_rag.uncertainty.threshold import AdaptiveThreshold
from harc_rag.routing.cost import CostEstimator


class AdaptiveRouter:

    # Keep verification context small enough for the verifier.
    MAX_VERIFICATION_CONTEXT_CHARS = 6000

    def __init__(self):
        self.threshold = AdaptiveThreshold()
        self.cost = CostEstimator()
        self.last_threshold = None

    def route(
        self,
        confidence: float,
        question: str = "",
        answer: str = "",
        context: str = "",
    ):

        threshold = self.threshold.calculate(question)
        self.last_threshold = threshold

        if confidence < threshold:

            return RoutingDecision(
                should_verify=True,
                confidence=confidence,
                reason="Low confidence; verification is required",
            )

        return RoutingDecision(
            should_verify=False,
            confidence=confidence,
            reason="Sufficient confidence; verification was not required",
        )
