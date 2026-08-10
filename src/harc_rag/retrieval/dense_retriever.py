from harc_rag.retrieval.interfaces import Retriever
from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.models import RetrievalResult
from harc_rag.vectorstore.exceptions import EmptyIndexError


class DenseRetriever(Retriever):

    def __init__(
        self,
        embedding_service,
        vector_store,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ):
        if self.embedding_service is None or self.vector_store is None:
            return []

        try:
            query_vector = self.embedding_service.model.embed_query(query)
            search_results = self.vector_store.search(query_vector, k)
        except EmptyIndexError:
            return []

        results = []

        for search_result in search_results:
            metadata = search_result.embedding.metadata

            results.append(
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id=search_result.embedding.chunk_id,
                        text=metadata.get("text", ""),
                        start_index=metadata.get("start_index", 0),
                        end_index=metadata.get("end_index", 0),
                        metadata={
                            key: value
                            for key, value in metadata.items()
                            if key not in {"text", "start_index", "end_index"}
                        },
                    ),
                    score=search_result.score,
                )
            )

        return results
