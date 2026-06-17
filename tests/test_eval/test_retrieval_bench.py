import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import pytest
from rag.eval.retrieval_bench import BenchmarkSample, run_retrieval_benchmark
from rag.ingest import Chunk
from rag.retrieve import bm25_retrieve

CORPUS = [
    Chunk("BM25 is a sparse retrieval algorithm using term frequency.", {"source": "t"}, "c1"),
    Chunk("Dense retrieval uses neural embeddings.", {"source": "t"}, "c2"),
    Chunk("Fine-tuning adapts pretrained models to specific tasks.", {"source": "t"}, "c3"),
]
SAMPLES = [BenchmarkSample(
    question="What is BM25?",
    answer="BM25 is a sparse retrieval algorithm.",
    supporting_facts=["BM25 is a sparse retrieval algorithm using term frequency."],
)]

def test_run_benchmark_returns_recall_dict():
    fn = lambda q, k: bm25_retrieve(q, CORPUS, k)
    metrics = run_retrieval_benchmark(SAMPLES, CORPUS, fn)
    assert "recall@3" in metrics and "recall@5" in metrics and "recall@10" in metrics

def test_recall_bounded():
    fn = lambda q, k: bm25_retrieve(q, CORPUS, k)
    metrics = run_retrieval_benchmark(SAMPLES, CORPUS, fn)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

def test_perfect_retrieval_gives_recall_one():
    fn = lambda q, k: bm25_retrieve(q, CORPUS, k)
    metrics = run_retrieval_benchmark(SAMPLES, CORPUS, fn, k_values=[3])
    assert metrics["recall@3"] == 1.0
