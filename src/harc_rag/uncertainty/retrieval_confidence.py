from harc_rag.retrieval.models import RetrievalResult


class RetrievalConfidenceEstimator:

    RRF_TOP_SCORE = 2 / 61

    def estimate(
        self,
        results: list[RetrievalResult],
    ) -> float:

        if not results:
            return 0.0

        scores = [
            result.score
            for result in results
        ]

        if self._looks_like_rrf(scores):
            scores = [
                score / self.RRF_TOP_SCORE
                for score in scores
            ]

        confidence = sum(scores) / len(scores)

        confidence = max(
            0.0,
            min(confidence, 1.0),
        )

        return confidence

    def _looks_like_rrf(
        self,
        scores: list[float],
    ) -> bool:

        return all(
            0.0 <= score <= self.RRF_TOP_SCORE
            for score in scores
        )
