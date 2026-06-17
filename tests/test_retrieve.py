import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from rag.ingest import Chunk
from rag.retrieve import RetrievalResult, bm25_retrieve, hybrid_rrf

CORPUS = [
    Chunk("RAG combines retrieval with language model generation.", {"topic": "RAG"}, "c1"),
    Chunk("BM25 is a sparse retrieval algorithm using term frequency.", {"topic": "retrieval"}, "c2"),
    Chunk("Dense retrieval uses neural embeddings for semantic search.", {"topic": "retrieval"}, "c3"),
    Chunk("Fine-tuning adapts a pretrained model to a specific task.", {"topic": "fine-tuning"}, "c4"),
    Chunk("RLHF trains models using human preference feedback.", {"topic": "alignment"}, "c5"),
]

def test_bm25_returns_results():
    results = bm25_retrieve("BM25 term frequency", CORPUS, k=3)
    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)

def test_bm25_ranks_relevant_first():
    results = bm25_retrieve("BM25 sparse retrieval", CORPUS, k=5)
    assert results[0].chunk_id == "c2"

def test_bm25_scores_descending():
    results = bm25_retrieve("retrieval", CORPUS, k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

def test_hybrid_rrf_deduplicates():
    bm25 = [RetrievalResult("c1", "text", {}, 1.0), RetrievalResult("c2", "text2", {}, 0.9)]
    dense = [RetrievalResult("c1", "text", {}, 0.95), RetrievalResult("c3", "text3", {}, 0.8)]
    fused = hybrid_rrf(bm25, dense)
    ids = [r.chunk_id for r in fused]
    assert len(ids) == len(set(ids))

def test_hybrid_rrf_scores_descending():
    bm25 = bm25_retrieve("retrieval", CORPUS, k=5)
    dense = [RetrievalResult("c1", CORPUS[0].text, {}, 0.9)]
    fused = hybrid_rrf(bm25, dense)
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)
