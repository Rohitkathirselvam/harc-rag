from harc_rag.chunking.models import Chunk
from harc_rag.retrieval.fusion import ReciprocalRankFusion
from harc_rag.retrieval.models import RetrievalResult


def test_rrf_combines_same_chunk_scores():

    chunk = Chunk(
        chunk_id=1,
        text="TCP",
        start_index=0,
        end_index=3,
        metadata={},
    )

    dense = [
        RetrievalResult(
            chunk=chunk,
            score=0.9,
        )
    ]

    bm25 = [
        RetrievalResult(
            chunk=chunk,
            score=7.0,
        )
    ]

    fusion = ReciprocalRankFusion(
        constant=60
    )

    results = fusion.fuse(
        dense,
        bm25,
    )

    assert len(results) == 1

    expected = (
        1 / 61
        + 1 / 61
    )

    assert abs(
        results[0].score - expected
    ) < 1e-9


def test_rrf_ranks_higher_fused_score_first():

    chunk1 = Chunk(
        chunk_id=1,
        text="TCP",
        start_index=0,
        end_index=3,
        metadata={},
    )

    chunk2 = Chunk(
        chunk_id=2,
        text="UDP",
        start_index=4,
        end_index=7,
        metadata={},
    )

    dense = [
        RetrievalResult(
            chunk=chunk1,
            score=0.9,
        ),
        RetrievalResult(
            chunk=chunk2,
            score=0.8,
        ),
    ]

    bm25 = [
        RetrievalResult(
            chunk=chunk1,
            score=7.0,
        )
    ]

    fusion = ReciprocalRankFusion(
        constant=60
    )

    results = fusion.fuse(
        dense,
        bm25,
    )

    assert len(results) == 2

    assert results[0].chunk.chunk_id == 1

    assert results[0].score > results[1].score