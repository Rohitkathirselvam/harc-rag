from rank_bm25 import BM25Okapi

from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult


class BM25Retriever:

    def __init__(self, chunks: list[Chunk]):

        self.chunks = chunks
        self._rebuild()

    def _rebuild(self):

        self.corpus = [
            chunk.text.split()
            for chunk in self.chunks
        ]

        self.bm25 = (
            BM25Okapi(self.corpus)
            if self.corpus
            else None
        )

    def add(self, chunks: list[Chunk]) -> None:

        self.chunks.extend(chunks)
        self._rebuild()

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievalResult]:

        if self.bm25 is None:
            return []

        query_tokens = query.split()

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for chunk, score in ranked[:k]:

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        return results
