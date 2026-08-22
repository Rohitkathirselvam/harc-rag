from harc_rag.retrieval.models import RetrievalResult


class RetrievalConfidenceEstimator:

    # Maximum possible RRF score when:
    # - two retrieval systems are used
    # - both return rank 1 for the same result
    #
    # 1 / (60 + 1) + 1 / (60 + 1)
    RRF_TOP_SCORE = 2 / 61

    def estimate(
        self,
        results: list[RetrievalResult],
    ) -> float:

        if not results:
            return 0.0

        scores = [
            max(float(result.score), 0.0)
            for result in results
        ]

        if self._looks_like_rrf(scores):
            confidence = self._rrf_confidence(scores)
        else:
            confidence = self._normal_score_confidence(
                scores
            )

        return max(
            0.0,
            min(float(confidence), 1.0),
        )

    # ---------------------------------------------------------
    # RRF CONFIDENCE
    # ---------------------------------------------------------

    def _rrf_confidence(
        self,
        scores: list[float],
    ) -> float:

        if not scores:
            return 0.0

        # Normalize against the theoretical maximum RRF
        # score instead of the maximum score in the current
        # result set.
        normalized_scores = [
            min(
                score / self.RRF_TOP_SCORE,
                1.0,
            )
            for score in scores
        ]

        # Average evidence strength across retrieved chunks.
        average_score = (
            sum(normalized_scores)
            / len(normalized_scores)
        )

        # The strongest retrieved result also matters.
        strongest_score = max(
            normalized_scores
        )

        # Give slightly more importance to the strongest
        # evidence while still considering all retrieved
        # results.
        confidence = (
            0.60 * strongest_score
            + 0.40 * average_score
        )

        return confidence

    # ---------------------------------------------------------
    # NORMALIZED RETRIEVAL CONFIDENCE
    # ---------------------------------------------------------

    def _normal_score_confidence(
        self,
        scores: list[float],
    ) -> float:

        if not scores:
            return 0.0

        normalized_scores = [
            min(max(score, 0.0), 1.0)
            for score in scores
        ]

        return (
            sum(normalized_scores)
            / len(normalized_scores)
        )

    # ---------------------------------------------------------
    # RRF DETECTION
    # ---------------------------------------------------------

    def _looks_like_rrf(
        self,
        scores: list[float],
    ) -> bool:

        if not scores:
            return False

        return (
            min(scores) >= 0.0
            and max(scores) <= self.RRF_TOP_SCORE
        )