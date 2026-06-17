from __future__ import annotations
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from FlagEmbedding import BGEM3FlagModel

from rag.ingest import Chunk

_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


class _BGE_M3(EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        out = _get_model().encode(input, batch_size=12, max_length=512)
        return out["dense_vecs"].tolist()


def embed_and_store(
    chunks: list[Chunk],
    collection_name: str,
    persist_dir: str = "data/chroma",
) -> chromadb.Collection:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(collection_name, embedding_function=_BGE_M3())
    existing = set(col.get()["ids"])
    new = [c for c in chunks if c.chunk_id not in existing]
    if new:
        col.add(
            ids=[c.chunk_id for c in new],
            documents=[c.text for c in new],
            metadatas=[c.metadata for c in new],
        )
    return col


def load_collection(
    collection_name: str,
    persist_dir: str = "data/chroma",
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name, embedding_function=_BGE_M3())
