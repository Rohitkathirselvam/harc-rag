from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import re
import string


class SemanticSimilarity:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = None

    def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:

        if not text1.strip() or not text2.strip():
            return 0.0

        try:
            score = self._claim_based_similarity(
                text1,
                text2,
            )
        except Exception:
            score = self._lexical_similarity(
                text1,
                text2,
            )

        return max(
            0.0,
            min(float(score), 1.0),
        )

    # ---------------------------------------------------------
    # CLAIM-BASED EVIDENCE SIMILARITY
    # ---------------------------------------------------------

    def _claim_based_similarity(
        self,
        answer: str,
        context: str,
    ) -> float:

        answer_sentences = self._split_sentences(answer)
        context_sentences = self._split_sentences(context)

        if not answer_sentences or not context_sentences:
            return 0.0

        if self.model is None:
            self.model = SentenceTransformer(
                self.model_name
            )

        answer_embeddings = self.model.encode(
            answer_sentences,
            convert_to_tensor=True,
        )

        context_embeddings = self.model.encode(
            context_sentences,
            convert_to_tensor=True,
        )

        similarities = cos_sim(
            answer_embeddings,
            context_embeddings,
        )

        claim_scores = []

        for row in similarities:
            best_score = float(row.max().item())

            claim_scores.append(
                max(
                    0.0,
                    min(best_score, 1.0),
                )
            )

        if not claim_scores:
            return 0.0

        # Average support across all answer claims.
        #
        # This is better than comparing the complete answer
        # against the complete context because every answer
        # sentence must find supporting evidence.
        return sum(claim_scores) / len(claim_scores)

    # ---------------------------------------------------------
    # SENTENCE SPLITTING
    # ---------------------------------------------------------

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ---------------------------------------------------------
    # LEXICAL FALLBACK
    # ---------------------------------------------------------

    def _lexical_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:

        sentences1 = self._split_sentences(text1)
        sentences2 = self._split_sentences(text2)

        if not sentences1 or not sentences2:
            return 0.0

        sentence_scores = []

        for sentence1 in sentences1:

            words1 = self._normalized_words(
                sentence1
            )

            if not words1:
                continue

            best_score = 0.0

            for sentence2 in sentences2:

                words2 = self._normalized_words(
                    sentence2
                )

                if not words2:
                    continue

                overlap = len(
                    words1 & words2
                ) / len(words1)

                best_score = max(
                    best_score,
                    overlap,
                )

            sentence_scores.append(
                min(best_score, 1.0)
            )

        if not sentence_scores:
            return 0.0

        return sum(sentence_scores) / len(
            sentence_scores
        )

    # ---------------------------------------------------------
    # WORD NORMALIZATION
    # ---------------------------------------------------------

    def _normalized_words(
        self,
        text: str,
    ) -> set[str]:

        translator = str.maketrans(
            "",
            "",
            string.punctuation,
        )

        words = (
            text.lower()
            .translate(translator)
            .split()
        )

        return {
            self._stem(word)
            for word in words
        }

    # ---------------------------------------------------------
    # SIMPLE STEMMING
    # ---------------------------------------------------------

    def _stem(
        self,
        word: str,
    ) -> str:

        if word == "using":
            return "use"

        for suffix in (
            "ing",
            "ed",
            "es",
            "s",
        ):

            if (
                len(word)
                > len(suffix) + 2
                and word.endswith(suffix)
            ):
                return word[:-len(suffix)]

        return word