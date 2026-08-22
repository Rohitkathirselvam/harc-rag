from harc_rag.chunking.exceptions import InvalidChunkConfigurationError
from harc_rag.chunking.models import Chunk
from harc_rag.chunking.strategies import ChunkingStrategy
from harc_rag.document.models import Document


class CharacterChunkingStrategy(ChunkingStrategy):

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise InvalidChunkConfigurationError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise InvalidChunkConfigurationError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigurationError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:

        text = document.text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        chunk_id = 0
        text_length = len(text)

        while start < text_length:

            target_end = min(
                start + self.chunk_size,
                text_length,
            )

            # If this is not the final chunk,
            # move the boundary to a nearby whitespace.
            if target_end < text_length:

                boundary = text.rfind(
                    " ",
                    start,
                    target_end,
                )

                if boundary > start:
                    end = boundary
                else:
                    end = target_end

            else:
                end = target_end

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_index=start,
                        end_index=end,
                        metadata={
                            "source": document.metadata.get(
                                "source"
                            )
                            if document.metadata
                            else None
                        },
                    )
                )

                chunk_id += 1

            # Prevent infinite loops
            if end >= text_length:
                break

            next_start = max(
                end - self.chunk_overlap,
                start + 1,
            )

            start = next_start

        return chunks