from sentence_transformers import SentenceTransformer

from harc_rag.chunking.models import Chunk
from harc_rag.embedding.interfaces import EmbeddingModel
from harc_rag.embedding.models import Embedding


class MiniLMEmbeddingModel(EmbeddingModel):

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, chunks: list[Chunk]) -> list[Embedding]:

        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(texts)

        embeddings = []

        for chunk, vector in zip(chunks, vectors):

            embeddings.append(
                Embedding(
                    chunk_id=chunk.chunk_id,
                    vector=vector.tolist(),
                    metadata={
                        **chunk.metadata.copy(),
                        "text": chunk.text,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                    },
                )
            )

        return embeddings

    def embed_query(self, query: str) -> list[float]:

        return self.model.encode([query])[0].tolist()
