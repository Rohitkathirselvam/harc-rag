class GenerationConfidenceEstimator:

    UNCERTAIN_PHRASES = [
        "i don't know",
        "i do not know",
        "not sure",
        "possibly",
        "maybe",
        "might",
        "could be",
        "cannot determine",
        "can't determine",
        "insufficient information",
        "not enough information",
        "unable to determine",
        "unclear",
    ]

    def estimate(self, answer: str) -> float:

        if not answer or not answer.strip():
            return 0.0

        answer_lower = answer.lower().strip()

        uncertainty_count = sum(
            1
            for phrase in self.UNCERTAIN_PHRASES
            if phrase in answer_lower
        )

        confidence = 1.0 - (
            0.15 * uncertainty_count
        )

        return max(
            0.0,
            min(confidence, 1.0),
        )