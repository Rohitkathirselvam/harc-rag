from pathlib import Path


class DocumentIndexingService:

    def __init__(
        self,
        loader,
        splitter,
        embedding_model,
        vector_store,
        bm25_retriever,
    ):
        self.loader = loader
        self.splitter = splitter
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def index(
        self,
        file_path: Path,
    ):

        # Load the PDF
        document = self.loader.load(file_path)

        # Split document into chunks
        chunks = self.splitter.split(document)

        # Generate embeddings
        embeddings = self.embedding_model.embed(chunks)

        # Add chunks to vector store
        self.vector_store.add(
            embeddings,
        )

        # Add chunks to BM25 index
        self.bm25_retriever.add(
            chunks
        )

        return chunks
