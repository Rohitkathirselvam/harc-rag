class AdaptiveThreshold:

    def __init__(self, default_threshold: float = 0.80):
        self.default_threshold = default_threshold

    def calculate(self, question: str = "") -> float:
        complexity_markers = {
            "compare",
            "differences",
            "explain",
            "relationship",
            "terms",
            "how",
            "why",
        }
        words = set(question.lower().split())
        if len(words) >= 10 or words & complexity_markers:
            return 0.60
        return self.default_threshold

    def needs_verification(self, confidence: float) -> bool:
        return confidence < self.default_threshold