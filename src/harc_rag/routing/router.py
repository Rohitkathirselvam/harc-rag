from harc_rag.routing.models import RoutingDecision
from harc_rag.uncertainty.threshold import AdaptiveThreshold
from harc_rag.routing.cost import CostEstimator


class AdaptiveRouter:

    def __init__(self):
        self.threshold = AdaptiveThreshold()
        self.cost = CostEstimator()

    def route(
        self,
        confidence: float,
        question: str = "",
        answer: str = "",
        context: str = "",
    ):

        threshold = self.threshold.calculate(
            question
        )

        if confidence < threshold:
            cost = self.cost.estimate(
                answer,
                context,
            )

            if cost < 150:

                return RoutingDecision(
                    should_verify=True,
                    confidence=confidence,
                    reason="Low confidence and affordable verification",
                )

            return RoutingDecision(
                should_verify=False,
                confidence=confidence,
                reason="Low confidence, but verification was skipped because the retrieved context is too large",
            )

        return RoutingDecision(
            should_verify=False,
            confidence=confidence,
            reason="Sufficient confidence; verification was not required",
        )       
