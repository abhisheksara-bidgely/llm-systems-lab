import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from rag.ingest import Chunk
from rag.embed import embed_and_store, load_collection

CHUNKS = [
    Chunk("Retrieval augmented generation combines retrieval with LLMs.", {"source_type": "paper", "topic_tag": "RAG"}, "id1"),
    Chunk("Dense retrieval uses vector embeddings for semantic search.", {"source_type": "paper", "topic_tag": "RAG"}, "id2"),
    Chunk("BM25 is a sparse term-frequency retrieval method.", {"source_type": "blog", "topic_tag": "RAG"}, "id3"),
]

def test_embed_stores_all_chunks(tmp_path):
    col = embed_and_store(CHUNKS, "test1", persist_dir=str(tmp_path))
    assert col.count() == 3

def test_stored_ids_match(tmp_path):
    col = embed_and_store(CHUNKS, "test2", persist_dir=str(tmp_path))
    assert set(col.get()["ids"]) == {"id1", "id2", "id3"}

def test_load_collection_works(tmp_path):
    embed_and_store(CHUNKS, "test3", persist_dir=str(tmp_path))
    col = load_collection("test3", persist_dir=str(tmp_path))
    assert col.count() == 3

def test_metadata_preserved(tmp_path):
    col = embed_and_store(CHUNKS, "test4", persist_dir=str(tmp_path))
    result = col.get(ids=["id3"], include=["metadatas"])
    assert result["metadatas"][0]["source_type"] == "blog"

def test_idempotent_store(tmp_path):
    embed_and_store(CHUNKS, "test5", persist_dir=str(tmp_path))
    embed_and_store(CHUNKS, "test5", persist_dir=str(tmp_path))
    col = load_collection("test5", persist_dir=str(tmp_path))
    assert col.count() == 3
