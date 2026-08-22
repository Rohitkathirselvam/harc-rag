class AdaptiveThreshold:

    def __init__(self, complexity_word_limit: int = 10):
        self.complexity_word_limit = complexity_word_limit

    def calculate(self, question: str = "") -> float:
        word_count = len(question.split())

        if word_count > self.complexity_word_limit:
            return 0.60

        return 0.80