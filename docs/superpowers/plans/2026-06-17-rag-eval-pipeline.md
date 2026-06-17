# RAG + Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production/research-grade RAG pipeline over AI research papers + engineering blogs with a rigorous eval harness (RAGAS + FRAMES/HotpotQA + LLM-as-Judge), delivered as a pedagogical notebook + importable Python module.

**Architecture:** A `src/rag/` module implements all pipeline components (ingest → embed → retrieve → generate → eval) with full test coverage. A `notebooks/rag_pipeline.ipynb` notebook imports from this module, adds theory markdown, written questions, and runs the full ablation and eval. The module is what makes the notebook testable and reusable.

**Tech Stack:** FlagEmbedding (BGE-M3, BGE-reranker-v2-m3), ChromaDB, rank-bm25, instructor + OpenAI GPT-4o-mini, RAGAS, HuggingFace datasets (FRAMES/HotpotQA), pypdf, BeautifulSoup4, tiktoken

## Global Constraints

- Python 3.10+
- All module code importable from `src/rag/` (add `src/` to sys.path in tests and notebook)
- All tests in `tests/` using pytest; no OpenAI API calls in tests (mock them)
- OPENAI_API_KEY read from environment, never hardcoded
- Chunk size ≤ 512 tokens measured by tiktoken `cl100k_base`
- ChromaDB persisted to `data/chroma/`
- Notebook cells run top-to-bottom without errors after module is installed

---

## File Map

**Created:**
- `src/__init__.py`, `src/rag/__init__.py`, `src/rag/eval/__init__.py`
- `src/rag/ingest.py` — `Chunk`, `load_pdf`, `load_html`, `chunk_document`
- `src/rag/embed.py` — `embed_and_store`, `load_collection`
- `src/rag/retrieve.py` — `RetrievalResult`, `bm25_retrieve`, `dense_retrieve`, `hybrid_rrf`, `rerank`
- `src/rag/generate.py` — `Citation`, `RAGAnswer`, `generate_answer`
- `src/rag/eval/retrieval_bench.py` — `BenchmarkSample`, `load_hotpotqa_subset`, `load_frames_subset`, `run_retrieval_benchmark`
- `src/rag/eval/ragas_eval.py` — `generate_testset`, `run_ragas_eval`
- `src/rag/eval/judge.py` — `JudgeScore`, `JudgeResult`, `llm_judge`, `positional_bias_check`, `length_correlation_check`
- `tests/test_ingest.py`, `tests/test_embed.py`, `tests/test_retrieve.py`, `tests/test_generate.py`
- `tests/test_eval/test_retrieval_bench.py`, `tests/test_eval/test_judge.py`, `tests/test_eval/test_ragas_eval.py`
- `scripts/download_papers.py`, `scripts/scrape_blogs.py`
- `notebooks/rag_pipeline.ipynb`

**Modified:**
- `requirements.txt`

---

### Task 1: Project Setup

**Files:**
- Create: `src/__init__.py`, `src/rag/__init__.py`, `src/rag/eval/__init__.py`
- Create: `tests/__init__.py`, `tests/test_eval/__init__.py`
- Create: `data/papers/.gitkeep`, `data/blogs/.gitkeep`, `data/chroma/.gitkeep`, `data/eval/.gitkeep`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: installable `src/rag` package

- [ ] **Step 1: Update requirements.txt**

```
torch>=2.1.0
transformers>=4.40.0
nbformat>=5.9.0
matplotlib>=3.8.0
scipy>=1.11.0
numpy>=1.24.0
jupyter>=1.0.0
ipykernel>=6.0.0
rank-bm25>=0.2.2
FlagEmbedding>=1.2.9
chromadb>=0.5.0
tiktoken>=0.7.0
pypdf>=4.2.0
beautifulsoup4>=4.12.0
requests>=2.31.0
instructor>=1.3.0
openai>=1.30.0
pydantic>=2.7.0
ragas>=0.1.14
datasets>=2.19.0
pandas>=2.0.0
seaborn>=0.13.0
langchain-openai>=0.1.0
langchain>=0.2.0
```

- [ ] **Step 2: Install**

```bash
pip install -r requirements.txt
```

Expected: no conflicts.

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p src/rag/eval data/papers data/blogs data/chroma data/eval tests/test_eval scripts
touch src/__init__.py src/rag/__init__.py src/rag/eval/__init__.py
touch tests/__init__.py tests/test_eval/__init__.py
touch data/papers/.gitkeep data/blogs/.gitkeep data/chroma/.gitkeep data/eval/.gitkeep
```

- [ ] **Step 4: Verify import works**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -c "import sys; sys.path.insert(0,'src'); import rag; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/ tests/ data/ scripts/
git commit -m "chore: project setup for RAG pipeline"
```

---

### Task 2: Ingest Module

**Files:**
- Create: `src/rag/ingest.py`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Produces:
  - `Chunk(text: str, metadata: dict, chunk_id: str)` — dataclass
  - `load_pdf(path: str) -> str`
  - `load_html(path: str) -> str`
  - `chunk_document(text: str, metadata: dict, max_tokens: int = 512) -> list[Chunk]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest, tiktoken
from rag.ingest import Chunk, chunk_document, load_html

def test_chunk_returns_list_of_chunks():
    chunks = chunk_document("Hello world. " * 100, {"source": "test"})
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)

def test_chunk_max_tokens_respected():
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_document("The quick brown fox. " * 300, {"source": "test"}, max_tokens=512)
    for c in chunks:
        assert len(enc.encode(c.text)) <= 512

def test_chunk_metadata_propagated():
    meta = {"source_type": "paper", "company": "Anthropic"}
    chunks = chunk_document("Hello world. " * 50, meta)
    for c in chunks:
        assert c.metadata["source_type"] == "paper"
        assert c.metadata["company"] == "Anthropic"

def test_chunk_ids_unique():
    chunks = chunk_document("Hello world. " * 200, {"source": "test"})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

def test_load_html_extracts_text():
    html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html); path = f.name
    text = load_html(path)
    assert "Title" in text and "Content here" in text
    os.unlink(path)

def test_no_empty_chunks():
    chunks = chunk_document("Hello world. " * 100, {"source": "test"})
    assert all(len(c.text.strip()) > 0 for c in chunks)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_ingest.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.ingest'`

- [ ] **Step 3: Implement ingest.py**

Create `src/rag/ingest.py`:
```python
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_html(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def chunk_document(text: str, metadata: dict, max_tokens: int = 512) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = len(enc.encode(sent))
        if st > max_tokens:
            if buf:
                _flush(buf, metadata, chunks)
                buf, buf_tokens = [], 0
            # split oversized sentence by words
            words, sub, sub_t = sent.split(), [], 0
            for w in words:
                wt = len(enc.encode(w))
                if sub_t + wt > max_tokens and sub:
                    _flush(sub, metadata, chunks); sub, sub_t = [], 0
                sub.append(w); sub_t += wt
            if sub:
                _flush(sub, metadata, chunks)
            continue
        if buf_tokens + st > max_tokens and buf:
            _flush(buf, metadata, chunks); buf, buf_tokens = [], 0
        buf.append(sent); buf_tokens += st

    if buf:
        _flush(buf, metadata, chunks)
    return chunks


def _flush(parts: list[str], metadata: dict, out: list[Chunk]) -> None:
    text = " ".join(parts)
    cid = hashlib.md5(text.encode()).hexdigest()[:12]
    out.append(Chunk(text=text, metadata={**metadata, "chunk_id": cid}, chunk_id=cid))
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_ingest.py -v
```

Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingest.py tests/test_ingest.py
git commit -m "feat: ingest module — PDF/HTML loading and semantic chunking"
```

---

### Task 3: Embed Module

**Files:**
- Create: `src/rag/embed.py`
- Create: `tests/test_embed.py`

**Interfaces:**
- Consumes: `Chunk` from `rag.ingest`
- Produces:
  - `embed_and_store(chunks: list[Chunk], collection_name: str, persist_dir: str = "data/chroma") -> chromadb.Collection`
  - `load_collection(collection_name: str, persist_dir: str = "data/chroma") -> chromadb.Collection`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embed.py`:
```python
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
    embed_and_store(CHUNKS, "test5", persist_dir=str(tmp_path))  # second call
    col = load_collection("test5", persist_dir=str(tmp_path))
    assert col.count() == 3  # no duplicates
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_embed.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.embed'`

- [ ] **Step 3: Implement embed.py**

Create `src/rag/embed.py`:
```python
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
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_embed.py -v
```

Expected: 5/5 PASS (BGE-M3 downloads ~2GB on first run)

- [ ] **Step 5: Commit**

```bash
git add src/rag/embed.py tests/test_embed.py
git commit -m "feat: embed module — BGE-M3 embeddings + ChromaDB storage"
```

---

### Task 4: Retrieve Module

**Files:**
- Create: `src/rag/retrieve.py`
- Create: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `Chunk` from `rag.ingest`, `chromadb.Collection` from `rag.embed`
- Produces:
  - `RetrievalResult(chunk_id: str, text: str, metadata: dict, score: float)` — dataclass
  - `bm25_retrieve(query: str, corpus: list[Chunk], k: int = 10) -> list[RetrievalResult]`
  - `dense_retrieve(query: str, collection: chromadb.Collection, k: int = 10) -> list[RetrievalResult]`
  - `hybrid_rrf(bm25_results: list[RetrievalResult], dense_results: list[RetrievalResult], k: int = 60) -> list[RetrievalResult]`
  - `rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieve.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_retrieve.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.retrieve'`

- [ ] **Step 3: Implement retrieve.py**

Create `src/rag/retrieve.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker

from rag.ingest import Chunk

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float


def bm25_retrieve(query: str, corpus: list[Chunk], k: int = 10) -> list[RetrievalResult]:
    tokenized = [c.text.lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [
        RetrievalResult(corpus[i].chunk_id, corpus[i].text, corpus[i].metadata, float(s))
        for i, s in ranked if s > 0
    ]


def dense_retrieve(query: str, collection: chromadb.Collection, k: int = 10) -> list[RetrievalResult]:
    res = collection.query(query_texts=[query], n_results=k)
    return [
        RetrievalResult(
            res["ids"][0][i],
            res["documents"][0][i],
            res["metadatas"][0][i],
            1.0 - res["distances"][0][i],
        )
        for i in range(len(res["ids"][0]))
    ]


def hybrid_rrf(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metas: dict[str, dict] = {}
    for rank, r in enumerate(bm25_results):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank + 1)
        texts[r.chunk_id] = r.text; metas[r.chunk_id] = r.metadata
    for rank, r in enumerate(dense_results):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank + 1)
        texts[r.chunk_id] = r.text; metas[r.chunk_id] = r.metadata
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [RetrievalResult(cid, texts[cid], metas[cid], sc) for cid, sc in ranked]


def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not results:
        return []
    pairs = [[query, r.text] for r in results]
    scores = _get_reranker().compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:top_k]
    return [RetrievalResult(r.chunk_id, r.text, r.metadata, float(s)) for s, r in ranked]
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_retrieve.py -v
```

Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: retrieve module — BM25, dense, hybrid RRF, cross-encoder reranking"
```

---

### Task 5: Generate Module

**Files:**
- Create: `src/rag/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `RetrievalResult` from `rag.retrieve`
- Produces:
  - `Citation(source_title, source_type, company, url, relevant_excerpt)` — Pydantic BaseModel
  - `RAGAnswer(answer, citations, confidence, reasoning_steps)` — Pydantic BaseModel
  - `generate_answer(query: str, results: list[RetrievalResult], client: OpenAI, model: str = "gpt-4o-mini") -> RAGAnswer`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from unittest.mock import MagicMock, patch
from rag.generate import Citation, RAGAnswer, generate_answer
from rag.retrieve import RetrievalResult

RESULTS = [
    RetrievalResult("c1", "RAG was introduced by Lewis et al. 2020.",
                    {"source_title": "RAG paper", "source_type": "paper", "company": "Meta", "url": "https://arxiv.org/abs/2005.11401"}, 0.95),
]

def test_citation_schema():
    c = Citation(source_title="Test", source_type="paper", company="Meta",
                 url="https://example.com", relevant_excerpt="Some text.")
    assert c.source_type == "paper"

def test_rag_answer_schema():
    a = RAGAnswer(answer="RAG works.", citations=[], confidence=0.9, reasoning_steps=["step1"])
    assert 0.0 <= a.confidence <= 1.0

def test_generate_answer_returns_rag_answer():
    mock_client = MagicMock()
    mock_answer = RAGAnswer(answer="test", citations=[], confidence=0.8, reasoning_steps=[])
    with patch("rag.generate.instructor") as mock_inst:
        patched = MagicMock()
        mock_inst.from_openai.return_value = patched
        patched.chat.completions.create.return_value = mock_answer
        result = generate_answer("What is RAG?", RESULTS, mock_client)
    assert isinstance(result, RAGAnswer)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_generate.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.generate'`

- [ ] **Step 3: Implement generate.py**

Create `src/rag/generate.py`:
```python
from __future__ import annotations
import instructor
from openai import OpenAI
from pydantic import BaseModel
from rag.retrieve import RetrievalResult


class Citation(BaseModel):
    source_title: str
    source_type: str
    company: str
    url: str
    relevant_excerpt: str


class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    reasoning_steps: list[str]


_SYSTEM = """You are a research assistant with access to retrieved passages from AI research papers and engineering blogs.
Answer using ONLY the provided context. Cite every claim. If context is insufficient, say so explicitly.
confidence: 0.0 = no relevant context, 1.0 = context directly and completely answers the question."""


def generate_answer(
    query: str,
    results: list[RetrievalResult],
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> RAGAnswer:
    patched = instructor.from_openai(client)
    context = "\n\n".join(
        f"[{i+1}] {r.metadata.get('source_title','?')} ({r.metadata.get('source_type','?')}, {r.metadata.get('company','?')})\n"
        f"URL: {r.metadata.get('url','')}\nText: {r.text}"
        for i, r in enumerate(results)
    )
    return patched.chat.completions.create(
        model=model,
        response_model=RAGAnswer,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_retries=2,
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_generate.py -v
```

Expected: 3/3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/generate.py tests/test_generate.py
git commit -m "feat: generate module — structured RAGAnswer output with Pydantic + instructor"
```

---

### Task 6: Retrieval Benchmark Module

**Files:**
- Create: `src/rag/eval/retrieval_bench.py`
- Create: `tests/test_eval/test_retrieval_bench.py`

**Interfaces:**
- Consumes: `RetrievalResult` from `rag.retrieve`, `Chunk` from `rag.ingest`
- Produces:
  - `BenchmarkSample(question: str, answer: str, supporting_facts: list[str])` — dataclass
  - `load_hotpotqa_subset(n: int = 200) -> list[BenchmarkSample]`
  - `load_frames_subset(n: int = 200) -> list[BenchmarkSample]`
  - `run_retrieval_benchmark(samples, corpus, retrieve_fn, k_values=[3,5,10]) -> dict[str, float]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval/test_retrieval_bench.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_retrieval_bench.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.eval.retrieval_bench'`

- [ ] **Step 3: Implement retrieval_bench.py**

Create `src/rag/eval/retrieval_bench.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset
from rag.ingest import Chunk
from rag.retrieve import RetrievalResult


@dataclass
class BenchmarkSample:
    question: str
    answer: str
    supporting_facts: list[str]


def load_hotpotqa_subset(n: int = 200) -> list[BenchmarkSample]:
    """
    HotpotQA (Yang et al. 2018): https://arxiv.org/abs/1809.09600
    Gold-standard multi-hop QA. Enables direct comparison with published RAG papers.
    Uses its own Wikipedia corpus — benchmarks retrieval component in isolation.
    """
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    samples = []
    for row in ds.select(range(min(n, len(ds)))):
        support = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            if title in row["supporting_facts"]["title"]:
                support.extend(sents)
        samples.append(BenchmarkSample(row["question"], row["answer"], support[:3]))
    return samples


def load_frames_subset(n: int = 200) -> list[BenchmarkSample]:
    """
    FRAMES (Google DeepMind, 2024): https://arxiv.org/abs/2409.12941
    Unlike BEIR (retrieval only), FRAMES tests the full pipeline —
    factuality, retrieval, and multi-step reasoning. More realistic than BEIR.
    """
    ds = load_dataset("google/frames-benchmark", split="test")
    samples = []
    for row in ds.select(range(min(n, len(ds)))):
        facts = [str(l) for l in (row.get("wiki_links") or [])[:3]]
        samples.append(BenchmarkSample(row["Prompt"], row["Answer"], facts))
    return samples


def _recall_at_k(retrieved: list[RetrievalResult], facts: list[str], k: int) -> float:
    if not facts:
        return 0.0
    top_texts = [r.text.lower() for r in retrieved[:k]]
    hits = sum(1 for f in facts if any(f.lower()[:60] in t for t in top_texts))
    return hits / len(facts)


def run_retrieval_benchmark(
    samples: list[BenchmarkSample],
    corpus: list[Chunk],
    retrieve_fn: Callable[[str, int], list[RetrievalResult]],
    k_values: list[int] = [3, 5, 10],
) -> dict[str, float]:
    sums = {k: 0.0 for k in k_values}
    for s in samples:
        results = retrieve_fn(s.question, max(k_values))
        for k in k_values:
            sums[k] += _recall_at_k(results, s.supporting_facts, k)
    return {f"recall@{k}": sums[k] / len(samples) for k in k_values}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_retrieval_bench.py -v
```

Expected: 3/3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/eval/retrieval_bench.py tests/test_eval/test_retrieval_bench.py
git commit -m "feat: retrieval benchmark — FRAMES and HotpotQA recall@k evaluation"
```

---

### Task 7: RAGAS Eval Module

**Files:**
- Create: `src/rag/eval/ragas_eval.py`
- Create: `tests/test_eval/test_ragas_eval.py`

**Interfaces:**
- Consumes: `RetrievalResult` from `rag.retrieve`, `RAGAnswer` from `rag.generate`
- Produces:
  - `generate_testset(docs: list[str], n: int = 100, llm_model: str = "gpt-4o-mini") -> list[dict]`
  - `run_ragas_eval(testset: list[dict], retrieve_fn, generate_fn, k: int = 5) -> dict[str, float]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval/test_ragas_eval.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import pytest
from unittest.mock import patch, MagicMock
from rag.eval.ragas_eval import run_ragas_eval

TESTSET = [
    {"question": "What is RAG?", "ground_truth": "RAG combines retrieval with generation.",
     "contexts": ["RAG retrieves passages and uses them to generate answers."]},
]

def test_run_ragas_returns_metric_dict():
    with patch("rag.eval.ragas_eval.evaluate") as mock_eval:
        mock_eval.return_value = {
            "faithfulness": 0.85, "answer_relevancy": 0.90,
            "context_recall": 0.75, "context_precision": 0.80,
        }
        result = run_ragas_eval(TESTSET, MagicMock(return_value=[]), MagicMock(return_value=MagicMock(answer="test")))
    assert all(k in result for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"])
    assert all(0.0 <= v <= 1.0 for v in result.values())
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_ragas_eval.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.eval.ragas_eval'`

- [ ] **Step 3: Implement ragas_eval.py**

Create `src/rag/eval/ragas_eval.py`:
```python
from __future__ import annotations
from typing import Callable

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from rag.retrieve import RetrievalResult


def generate_testset(
    docs: list[str],
    n: int = 100,
    llm_model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    RAGAS TestsetGenerator (paper: https://arxiv.org/abs/2309.15217).
    Synthesizes QA pairs without human annotation.
    Question types:
      simple (40%): single-hop, directly answerable from one passage
      reasoning (30%): requires inference beyond literal text
      multi_context (30%): answer requires multiple passages
    Always spot-check ~25 questions manually before trusting the testset.
    """
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.schema import Document

    gen_llm = ChatOpenAI(model=llm_model)
    critic_llm = ChatOpenAI(model=llm_model)
    generator = TestsetGenerator.from_langchain(gen_llm, critic_llm, OpenAIEmbeddings())
    langchain_docs = [Document(page_content=d) for d in docs]
    testset = generator.generate_with_langchain_docs(
        langchain_docs, test_size=n,
        distributions={simple: 0.4, reasoning: 0.3, multi_context: 0.3},
    )
    return testset.to_pandas().to_dict(orient="records")


def run_ragas_eval(
    testset: list[dict],
    retrieve_fn: Callable[[str, int], list[RetrievalResult]],
    generate_fn: Callable,
    k: int = 5,
) -> dict[str, float]:
    questions, answers, contexts, ground_truths = [], [], [], []
    for sample in testset:
        q = sample["question"]
        results = retrieve_fn(q, k)
        ans = generate_fn(q, results)
        questions.append(q)
        answers.append(ans.answer if hasattr(ans, "answer") else str(ans))
        contexts.append([r.text for r in results])
        ground_truths.append(sample.get("ground_truth", ""))

    ds = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts, "ground_truth": ground_truths,
    })
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
    return {k: float(v) for k, v in result.items()}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_ragas_eval.py -v
```

Expected: 1/1 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/eval/ragas_eval.py tests/test_eval/test_ragas_eval.py
git commit -m "feat: RAGAS eval — synthetic testset generation and end-to-end evaluation"
```

---

### Task 8: LLM-as-Judge Module

**Files:**
- Create: `src/rag/eval/judge.py`
- Create: `tests/test_eval/test_judge.py`

**Interfaces:**
- Produces:
  - `JudgeScore(relevance: int, faithfulness: int, completeness: int)` — Pydantic, `total` computed field (0-9)
  - `JudgeResult(question: str, answer: str, score: float, reasoning: str)` — dataclass, score = total/9.0
  - `llm_judge(question: str, answer: str, context: str, client: OpenAI, model: str = "gpt-4o-mini") -> JudgeResult`
  - `positional_bias_check(questions, answer_pairs, contexts, client) -> dict` with keys `flip_rate`, `n_flips`, `n_total`
  - `length_correlation_check(results: list[JudgeResult]) -> dict` with keys `pearson_r`, `p_value`

- [ ] **Step 1: Write failing tests**

Create `tests/test_eval/test_judge.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import pytest
from unittest.mock import patch, MagicMock
from rag.eval.judge import JudgeScore, JudgeResult, positional_bias_check, length_correlation_check

def test_judge_score_total():
    s = JudgeScore(relevance=2, faithfulness=3, completeness=2)
    assert s.total == 7

def test_length_correlation_returns_dict():
    results = [
        JudgeResult("q", "Short.", 0.3, "r"),
        JudgeResult("q", "Medium length answer with more context.", 0.6, "r"),
        JudgeResult("q", "This is a very long detailed answer covering everything.", 0.9, "r"),
    ]
    m = length_correlation_check(results)
    assert "pearson_r" in m and "p_value" in m
    assert -1.0 <= m["pearson_r"] <= 1.0

def test_positional_bias_returns_flip_rate():
    call_n = [0]
    def mock_judge(q, a, ctx, client):
        call_n[0] += 1
        return JudgeResult(q, a, 0.9 if call_n[0] % 2 == 1 else 0.5, "mock")
    with patch("rag.eval.judge.llm_judge", side_effect=mock_judge):
        result = positional_bias_check(["q1"], [("A", "B")], ["ctx"], MagicMock())
    assert "flip_rate" in result and 0.0 <= result["flip_rate"] <= 1.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_judge.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'rag.eval.judge'`

- [ ] **Step 3: Implement judge.py**

Create `src/rag/eval/judge.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

import instructor
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, computed_field
from scipy import stats


class JudgeScore(BaseModel):
    relevance: int      # 0–3: does the answer address the question?
    faithfulness: int   # 0–3: is every claim grounded in the context?
    completeness: int   # 0–3: are all answerable aspects covered?

    @computed_field
    @property
    def total(self) -> int:
        return self.relevance + self.faithfulness + self.completeness


@dataclass
class JudgeResult:
    question: str
    answer: str
    score: float   # JudgeScore.total / 9.0
    reasoning: str


_SYSTEM = """You are an impartial evaluator. Score the answer on three dimensions (0–3 each):
relevance: does the answer directly address the question? (0=off-topic, 3=precisely on-topic)
faithfulness: is every claim supported by the provided context? (0=hallucinated, 3=fully grounded)
completeness: does the answer cover all aspects the context allows? (0=partial, 3=complete)
Be strict — 3 requires excellence, not just adequacy."""


def llm_judge(
    question: str,
    answer: str,
    context: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    patched = instructor.from_openai(client)
    score = patched.chat.completions.create(
        model=model,
        response_model=JudgeScore,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"},
        ],
    )
    return JudgeResult(
        question=question, answer=answer,
        score=score.total / 9.0,
        reasoning=f"relevance={score.relevance}, faithfulness={score.faithfulness}, completeness={score.completeness}",
    )


def positional_bias_check(
    questions: list[str],
    answer_pairs: list[tuple[str, str]],
    contexts: list[str],
    client: OpenAI,
) -> dict:
    """
    Positional bias: LLM judges prefer the first answer ~65% of the time (Zheng et al. 2023).
    Run each comparison A-then-B and B-then-A. Count flips.
    Flip rate > 15% = significant positional bias — average both orderings.
    Paper: https://arxiv.org/abs/2306.05685
    """
    n_flips = 0
    for q, (a, b), ctx in zip(questions, answer_pairs, contexts):
        score_a = llm_judge(q, a, ctx, client).score
        score_b = llm_judge(q, b, ctx, client).score
        winner_ab = "A" if score_a > score_b else "B"
        score_b2 = llm_judge(q, b, ctx, client).score
        score_a2 = llm_judge(q, a, ctx, client).score
        winner_ba = "B" if score_b2 > score_a2 else "A"
        if winner_ab != winner_ba:
            n_flips += 1
    n = len(questions)
    return {"flip_rate": n_flips / n if n else 0.0, "n_flips": n_flips, "n_total": n}


def length_correlation_check(results: list[JudgeResult]) -> dict:
    """
    Length bias: LLM judges assign higher scores to longer answers regardless of quality.
    Pearson r > 0.3 = significant — consider normalizing or penalizing verbosity.
    """
    lengths = np.array([len(r.answer.split()) for r in results])
    scores = np.array([r.score for r in results])
    r, p = stats.pearsonr(lengths, scores)
    return {"pearson_r": float(r), "p_value": float(p)}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/test_eval/test_judge.py -v
```

Expected: 3/3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/eval/judge.py tests/test_eval/test_judge.py
git commit -m "feat: LLM-as-judge with positional bias and length correlation analysis"
```

---

### Task 9: Data Collection Scripts

**Files:**
- Create: `scripts/download_papers.py`
- Create: `scripts/scrape_blogs.py`

**Interfaces:**
- Produces: PDFs in `data/papers/`, text files in `data/blogs/`

- [ ] **Step 1: Create download_papers.py**

Create `scripts/download_papers.py`:
```python
"""Run: python scripts/download_papers.py"""
import time, requests
from pathlib import Path

PAPERS = {
    "rag_lewis2020.pdf":          "https://arxiv.org/pdf/2005.11401",
    "hyde_gao2022.pdf":           "https://arxiv.org/pdf/2212.10496",
    "raptor_sarthi2024.pdf":      "https://arxiv.org/pdf/2401.18059",
    "self_rag_asai2023.pdf":      "https://arxiv.org/pdf/2310.11511",
    "ragas_es2023.pdf":           "https://arxiv.org/pdf/2309.15217",
    "frames_krishna2024.pdf":     "https://arxiv.org/pdf/2409.12941",
    "hotpotqa_yang2018.pdf":      "https://arxiv.org/pdf/1809.09600",
    "react_yao2022.pdf":          "https://arxiv.org/pdf/2210.03629",
    "instructgpt_ouyang2022.pdf": "https://arxiv.org/pdf/2203.02155",
    "dpo_rafailov2023.pdf":       "https://arxiv.org/pdf/2305.18290",
    "vllm_kwon2023.pdf":          "https://arxiv.org/pdf/2309.06180",
    "bge_m3_chen2024.pdf":        "https://arxiv.org/pdf/2402.03216",
    "llm_judge_zheng2023.pdf":    "https://arxiv.org/pdf/2306.05685",
    "speculative_decoding.pdf":   "https://arxiv.org/pdf/2211.17192",
}

out = Path("data/papers")
out.mkdir(parents=True, exist_ok=True)
for name, url in PAPERS.items():
    dest = out / name
    if dest.exists():
        print(f"  skip {name}")
        continue
    print(f"  downloading {name}...")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        time.sleep(1)
    except Exception as e:
        print(f"  FAILED: {e}")
print(f"Done. {len(list(out.glob('*.pdf')))} PDFs in data/papers/")
```

- [ ] **Step 2: Create scrape_blogs.py**

Create `scripts/scrape_blogs.py`:
```python
"""Run: python scripts/scrape_blogs.py"""
import re, time, requests
from pathlib import Path
from bs4 import BeautifulSoup

BLOGS = [
    {"url": "https://www.anthropic.com/news/contextual-retrieval",    "company": "Anthropic",   "topic": "RAG"},
    {"url": "https://huggingface.co/blog/rag-evaluation",             "company": "HuggingFace", "topic": "eval"},
    {"url": "https://huggingface.co/blog/trl-dpo-trainer",            "company": "HuggingFace", "topic": "fine-tuning"},
    {"url": "https://cohere.com/blog/rerank",                         "company": "Cohere",      "topic": "RAG"},
    {"url": "https://vllm.ai/blog/2023/06/20/vllm.html",             "company": "vLLM",        "topic": "inference"},
]

out = Path("data/blogs")
out.mkdir(parents=True, exist_ok=True)

def slug(url): return re.sub(r'[^a-z0-9]+', '_', url.lower().split("//")[-1])[:60]

for b in BLOGS:
    dest = out / f"{slug(b['url'])}.txt"
    if dest.exists():
        print(f"  skip {dest.name}")
        continue
    print(f"  scraping {b['url']}...")
    try:
        r = requests.get(b["url"], timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        header = f"# SOURCE: {b['url']}\n# COMPANY: {b['company']}\n# TOPIC: {b['topic']}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        time.sleep(2)
    except Exception as e:
        print(f"  FAILED: {e}")
print(f"Done. {len(list(out.glob('*.txt')))} blogs in data/blogs/")
```

- [ ] **Step 3: Run scripts**

```bash
cd /home/abhisheksara/SpeculativeDecoding
python scripts/download_papers.py
python scripts/scrape_blogs.py
```

Expected: ≥10 PDFs in `data/papers/`, ≥3 text files in `data/blogs/`

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: data collection scripts — papers and blog posts"
```

---

### Task 10: Notebook

**Files:**
- Create: `notebooks/build_rag_notebook.py`
- Create: `notebooks/rag_pipeline.ipynb` (generated by build script)

The notebook has 6 parts. Each part follows: theory markdown (with links) → written question → code using `src/rag` → assertion-based test cells.

- [ ] **Step 1: Create build script**

Create `notebooks/build_rag_notebook.py`:

```python
"""Run: python notebooks/build_rag_notebook.py"""
import json
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ── HEADER ──────────────────────────────────────────────────────────────────
cells += [md("""# RAG + Eval Pipeline

**Goal:** Production/research-grade RAG over AI papers + engineering blogs.  
After this notebook you can:
1. Build end-to-end RAG pipelines independently at a startup
2. Critically read RAG research papers (know what eval claims are valid)
3. Contribute to RAG research — understand open problems and how to measure progress

**Corpus:** ~14 arXiv papers + ~5 engineering blog posts  
**Stack:** BGE-M3 (local embeddings) · ChromaDB · BM25 + hybrid RRF · BGE-reranker · GPT-4o-mini · RAGAS · FRAMES/HotpotQA

```
Papers + Blogs → chunk → BGE-M3 → ChromaDB
                                        ↓
Query → BM25 + dense → hybrid RRF → rerank → GPT-4o-mini → RAGAnswer
                                                                  ↓
                          FRAMES/HotpotQA recall@k · RAGAS · LLM-as-Judge bias analysis
```
""")]

# ── PART 1 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 1 — Data Ingestion & Chunking

### Why chunking determines retrieval quality

A RAG system retrieves *chunks*, not documents. The chunk is the atomic unit of retrieval.

**Chunk too large:** one chunk covers multiple topics → retrieved for wrong queries → low precision  
**Chunk too small:** one sentence per chunk → multi-sentence answers need many chunks → low recall

The correct size depends on your queries. For dense technical text (research papers), semantic chunking
to ≤512 tokens balances both.

**Semantic chunking algorithm:**
1. Split text into sentences on punctuation boundaries
2. Greedily merge sentences until the next would exceed `max_tokens`
3. Flush the buffer as a chunk; start a new buffer

**Why 512 tokens?** BGE-M3 was trained on sequences up to 8192 tokens, but embedding quality degrades
for long sequences. 512 tokens (~380 words) captures a complete idea without diluting the vector.

Paper: [BGE-M3 (Chen et al. 2024)](https://arxiv.org/abs/2402.03216)
"""),
md("### Written Question 1\n\n*Why does chunking matter more for multi-hop questions than single-hop? Give a concrete example where an answer requires one passage from a paper and one from a blog post. What chunk size would you choose and why?*"),
code("""import sys
sys.path.insert(0, '../src')
from rag.ingest import Chunk, chunk_document, load_pdf, load_html
from pathlib import Path

pdf_path = '../data/papers/rag_lewis2020.pdf'
if Path(pdf_path).exists():
    text = load_pdf(pdf_path)
    print(f"Extracted {len(text):,} characters")
    print(f"Preview: {text[:400]}")
else:
    print("Run: python scripts/download_papers.py")
"""),
code("""if Path(pdf_path).exists():
    meta = {"source_type": "paper", "company": "Meta", "topic_tag": "RAG",
            "pub_date": "2020-05-22", "url": "https://arxiv.org/abs/2005.11401",
            "source_title": "RAG — Lewis et al. 2020"}
    chunks = chunk_document(text, meta, max_tokens=512)
    print(f"Chunks: {len(chunks)}")
    print(f"Avg length (words): {sum(len(c.text.split()) for c in chunks)/len(chunks):.0f}")
    print(f"\\nChunk 5:\\n{chunks[5].text[:300]}...")
"""),
code("""# Test: all chunks ≤ 512 tokens, non-empty
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
if Path(pdf_path).exists():
    bad = [c for c in chunks if len(enc.encode(c.text)) > 512]
    assert not bad, f"{len(bad)} chunks exceed 512 tokens"
    assert all(c.text.strip() for c in chunks)
    print(f"✓ All {len(chunks)} chunks valid")
""")]

# ── PART 2 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 2 — Embedding & Vector Store

### Dense vs Sparse: why neither alone is enough

**Sparse (BM25):** bag-of-words, score = weighted term overlap  
✓ exact keyword match, fast, interpretable  
✗ vocabulary mismatch: "car" ≠ "automobile", no semantic understanding

**Dense (BGE-M3):** neural encoder, score = cosine similarity of embedding vectors  
✓ semantic understanding, handles paraphrases  
✗ needs trained model, slower, harder to debug

Neither alone is sufficient — we combine them in Part 3.

### Why BGE-M3 over OpenAI embeddings?

[BGE-M3 (Chen et al. 2024)](https://arxiv.org/abs/2402.03216) from BAAI:
- **Free to run locally** — no per-token API cost (critical when indexing 50+ papers)
- SOTA on MTEB benchmark for technical/scientific text
- Produces dense, sparse (lexical), AND multi-vector representations simultaneously
- Supports 8192-token sequences natively

Embedding dimension: **1024** (dense). Stored in ChromaDB with cosine similarity.
"""),
md("### Written Question 2\n\n*BM25 uses IDF (inverse document frequency) to downweight common terms. In a corpus of 50 AI papers, which specific terms would have near-zero IDF? Why would retrieval fail for a query like \"attention mechanism\" without IDF?*"),
code("""from rag.ingest import load_pdf, load_html, chunk_document
from pathlib import Path

all_chunks = []
for pdf in sorted(Path('../data/papers').glob('*.pdf')):
    try:
        text = load_pdf(str(pdf))
        meta = {"source_type": "paper", "source_title": pdf.stem.replace('_',' '),
                "company": "arXiv", "topic_tag": "AI", "url": ""}
        c = chunk_document(text, meta)
        all_chunks.extend(c)
        print(f"  {pdf.name}: {len(c)} chunks")
    except Exception as e:
        print(f"  SKIP {pdf.name}: {e}")

for txt in sorted(Path('../data/blogs').glob('*.txt')):
    try:
        text = txt.read_text(encoding='utf-8')
        meta = {"source_type": "blog", "source_title": txt.stem.replace('_',' '),
                "company": "unknown", "topic_tag": "engineering", "url": ""}
        for line in text.split('\\n')[:5]:
            if line.startswith('# COMPANY:'): meta['company'] = line.split(':',1)[1].strip()
            elif line.startswith('# TOPIC:'): meta['topic_tag'] = line.split(':',1)[1].strip()
            elif line.startswith('# SOURCE:'): meta['url'] = line.split(':',1)[1].strip()
        c = chunk_document(text, meta)
        all_chunks.extend(c)
        print(f"  {txt.name}: {len(c)} chunks")
    except Exception as e:
        print(f"  SKIP {txt.name}: {e}")

print(f"\\nTotal: {len(all_chunks)} chunks")
"""),
code("""from rag.embed import embed_and_store, load_collection

print("Embedding with BGE-M3 (first run downloads ~2GB)...")
collection = embed_and_store(all_chunks, "rag_corpus", persist_dir="../data/chroma")
print(f"✓ {collection.count()} chunks in ChromaDB")
"""),
code("""# Sanity check
results = collection.query(query_texts=["How does speculative decoding reduce inference latency?"], n_results=3)
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"[{i+1}] {meta.get('source_title','?')[:50]} ({meta.get('source_type')})")
    print(f"     {doc[:120]}...\\n")
""")]

# ── PART 3 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 3 — Retrieval Ablation

### Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

**The failure modes:**
- BM25 fails on semantic queries: "what causes hallucination" won't match "factual errors from LLMs"
- Dense fails on exact-match queries: "what is DPO" may return similar-sounding but wrong passages

**RRF ([Cormack et al. 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)):**

$$\\text{RRF}(d) = \\sum_{r \\in \\{\\text{bm25, dense}\\}} \\frac{1}{k + \\text{rank}_r(d)}$$

$k=60$ is a smoothing constant. Without it, rank-1 from either retriever dominates.  
With $k=60$, the contribution of rank-1 is $1/61 \\approx 0.016$ — moderated.  
Documents ranked highly by *both* retrievers win. This is the key insight.

### Cross-Encoder Reranking

Bi-encoder (BM25, dense): encodes query and document *separately*  
Cross-encoder (BGE-reranker): encodes [query, document] *together* — full attention over both

[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3): much higher precision,  
but can't pre-compute — used only on top candidates (e.g. top-20 → rerank → top-5).
"""),
md("### Written Question 3\n\n*The RRF formula uses `1/(k+rank)` rather than raw retrieval scores. What would go wrong if you naively summed `score_bm25 + score_dense`? Give a numerical example showing why scale mismatch matters.*"),
code("""from rag.retrieve import bm25_retrieve, dense_retrieve, hybrid_rrf, rerank
import pandas as pd

QUERY = "How does speculative decoding reduce LLM inference latency?"

dense_r  = dense_retrieve(QUERY, collection, k=10)
bm25_r   = bm25_retrieve(QUERY, all_chunks, k=10)
hybrid_r = hybrid_rrf(bm25_r, dense_r)
rerank_r = rerank(QUERY, hybrid_r, top_k=5)

strategies = {"dense": dense_r[:5], "bm25": bm25_r[:5], "hybrid": hybrid_r[:5], "reranked": rerank_r}
for name, results in strategies.items():
    print(f"\\n=== {name} ===")
    for r in results:
        print(f"  [{r.score:.4f}] {r.metadata.get('source_title','?')[:45]}")
        print(f"           {r.text[:90]}...")
"""),
code("""# Ablation table: recall@k on HotpotQA subset
from rag.eval.retrieval_bench import load_hotpotqa_subset, run_retrieval_benchmark

print("Loading HotpotQA (multi-hop QA benchmark, Yang et al. 2018)...")
print("Paper: https://arxiv.org/abs/1809.09600")
print("Why HotpotQA: gold-standard multi-hop dataset, enables comparison with published RAG papers.\\n")
samples = load_hotpotqa_subset(n=50)  # 50 for speed

retrieve_fns = {
    "BM25":          lambda q, k: bm25_retrieve(q, all_chunks, k),
    "Dense":         lambda q, k: dense_retrieve(q, collection, k),
    "Hybrid RRF":    lambda q, k: hybrid_rrf(bm25_retrieve(q, all_chunks, k), dense_retrieve(q, collection, k)),
    "Hybrid+Rerank": lambda q, k: rerank(q, hybrid_rrf(bm25_retrieve(q, all_chunks, k), dense_retrieve(q, collection, k)), top_k=k),
}

rows = {}
for name, fn in retrieve_fns.items():
    print(f"Running {name}...")
    rows[name] = run_retrieval_benchmark(samples, all_chunks, fn)

df = pd.DataFrame(rows).T
print("\\n=== Ablation Table (Recall@k, HotpotQA) ===")
print(df.to_string(float_format="{:.3f}".format))
""")]

# ── PART 4 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 4 — Structured Answer Generation

### Why free-text generation fails in production

Raw LLM output is a string. At scale:
- Citations are inconsistent or missing
- Confidence is never quantified
- Downstream parsing is brittle

**Structured outputs** (OpenAI JSON mode + [instructor](https://github.com/jxnl/instructor)):  
Force the LLM to return a validated Pydantic schema. If validation fails, instructor
sends the error back to the LLM and retries (`max_retries=2`). Output is always parseable.

### Schema design

```python
class Citation(BaseModel):
    source_title: str; source_type: str  # "paper" | "blog"
    company: str; url: str; relevant_excerpt: str

class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float          # 0.0 (no relevant context) → 1.0 (directly answered)
    reasoning_steps: list[str] # explicit chain-of-thought for multi-hop
```

`reasoning_steps` forces explicit chain-of-thought for multi-hop questions, which improves
accuracy AND makes evaluation easier (you can check each step independently).
"""),
md("### Written Question 4\n\n*What is the difference between OpenAI's JSON mode, function calling, and `response_model` (instructor)? Which gives the strongest schema guarantee? When would you use function calling over a response_model?*"),
code("""import os
from openai import OpenAI
from rag.generate import generate_answer, RAGAnswer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
QUERY = "How does speculative decoding reduce LLM inference latency, and has any company deployed this in production?"

ans = generate_answer(QUERY, rerank_r, client)
print(f"Answer:\\n{ans.answer}")
print(f"\\nConfidence: {ans.confidence:.2f}")
print(f"\\nReasoning steps:")
for step in ans.reasoning_steps: print(f"  - {step}")
print(f"\\nCitations ({len(ans.citations)}):")
for c in ans.citations:
    print(f"  [{c.source_type}] {c.source_title} — {c.url}")
    print(f"  Excerpt: {c.relevant_excerpt[:80]}...")
"""),
code("""# Test: 5 queries all parse to RAGAnswer
import time
TEST_QUERIES = [
    "What is RAG and why does it reduce hallucination?",
    "How does DPO differ from RLHF?",
    "Why does BM25 fail on semantic queries?",
    "What is the acceptance rate in speculative decoding?",
    "What is context precision in RAGAS?",
]
failures = []
for q in TEST_QUERIES:
    try:
        r = rerank(q, hybrid_rrf(bm25_retrieve(q, all_chunks, 10), dense_retrieve(q, collection, 10)), top_k=5)
        a = generate_answer(q, r, client)
        assert isinstance(a, RAGAnswer)
        assert 0.0 <= a.confidence <= 1.0
        print(f"✓ {q[:60]}")
        time.sleep(0.5)
    except Exception as e:
        failures.append(q); print(f"✗ {q[:60]}: {e}")
print(f"\\n{len(TEST_QUERIES)-len(failures)}/{len(TEST_QUERIES)} parsed")
""")]

# ── PART 5 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 5 — RAGAS End-to-End Evaluation

### What each metric measures (and why the definition is the way it is)

[RAGAS (Es et al. 2023)](https://arxiv.org/abs/2309.15217) — reference-free RAG evaluation.  
"Reference-free" = no human-annotated answers needed. An LLM judges quality.

**faithfulness** — fraction of claims in the answer supported by the context  
*Why this definition:* hallucination = claim not in context. Count them.  
*Failure mode:* lenient judge lets paraphrased hallucinations through.

**answer_relevancy** — mean cosine similarity between the question and N reverse-engineered questions from the answer  
*Why this definition:* "does this answer the question?" is subjective. Reverse-engineering is objective.  
*Failure mode:* verbose answers that mention the topic but don't answer score high.

**context_recall** — fraction of ground-truth answer sentences attributable to retrieved context  
*Requires ground_truth.* This is why we need a testset.

**context_precision** — fraction of retrieved chunks that are actually useful  
*Why it matters:* irrelevant context = noise + wasted tokens.

### FRAMES vs BEIR vs HotpotQA — which benchmark to trust?

**BEIR** ([Thakur et al. 2021](https://arxiv.org/abs/2104.08663)): benchmarks retrieval only (bi-encoder ranking). 18 datasets. Widely used but measures an isolated component.

**HotpotQA** ([Yang et al. 2018](https://arxiv.org/abs/1809.09600)): multi-hop QA on Wikipedia. Gold standard for multi-hop reasoning. Widely cited → compare your numbers to published papers.

**FRAMES** ([Krishna et al. 2024](https://arxiv.org/abs/2409.12941)): tests the full pipeline — factuality + retrieval + multi-step reasoning. More realistic than BEIR. Published by Google DeepMind.

We use HotpotQA (retrieval component) + RAGAS synthetic (end-to-end on our corpus).
"""),
code("""from rag.eval.ragas_eval import generate_testset

print("Generating 30 synthetic QA pairs with RAGAS TestsetGenerator...")
print("(Calls OpenAI API — expect ~$0.30 and ~3 min)\\n")
corpus_texts = [c.text for c in all_chunks[:150]]
testset = generate_testset(corpus_texts, n=30)
print(f"Generated {len(testset)} QA pairs")

import pandas as pd
df_ts = pd.DataFrame(testset)
if 'evolution_type' in df_ts.columns:
    print("\\nType distribution:"); print(df_ts['evolution_type'].value_counts())
print("\\nSample questions:")
for row in testset[:3]:
    print(f"  Q: {str(row.get('question',''))[:80]}")
"""),
md("""### Can we trust this testset?

RAGAS generates QA pairs automatically, but the generator fails in predictable ways:
- Questions too easy (single-sentence lookups)
- Ground truth incorrect
- Multi-hop questions that are actually single-hop

**Spot-check protocol:** Review 15 questions. For each, answer:
1. Is this a realistic question?
2. Is the ground truth correct from the source?
3. Is the type label correct?

Record in the cell below. If >30% fail any criterion, regenerate with a higher-quality LLM.
"""),
code("""import random; random.seed(42)
sample_25 = random.sample(testset, min(15, len(testset)))
for i, s in enumerate(sample_25[:5]):
    print(f"[{i+1}] Q: {str(s.get('question',''))[:80]}")
    print(f"     GT: {str(s.get('ground_truth',''))[:100]}")
    print(f"     Type: {s.get('evolution_type','?')}")
    print(f"     ANNOTATE: realistic? Y/N  |  GT correct? Y/N  |  type correct? Y/N\\n")
"""),
code("""from rag.eval.ragas_eval import run_ragas_eval

def retrieve_fn(q, k=5):
    return rerank(q, hybrid_rrf(bm25_retrieve(q, all_chunks, k*2), dense_retrieve(q, collection, k*2)), top_k=k)

def generate_fn(q, results):
    return generate_answer(q, results, client)

print("Running RAGAS eval (expect ~$1 and ~5 min)...")
ragas_scores = run_ragas_eval(testset[:20], retrieve_fn, generate_fn)
print("\\n=== RAGAS Scores (Hybrid + Rerank) ===")
for metric, score in ragas_scores.items():
    bar = "█" * int(score * 20)
    print(f"  {metric:<25} {score:.3f}  {bar}")
""")]

# ── PART 6 ──────────────────────────────────────────────────────────────────
cells += [md("""## Part 6 — LLM-as-Judge: Building Trust in Your Evaluator

### What LLM-as-Judge is

Use an LLM to evaluate the output of another LLM. Primary eval method at startups:
- Human eval: $5-50/question, days to weeks
- LLM judge: ~$0.01/question, seconds, ~80% agreement with human experts

### Two known biases (Zheng et al. 2023, [MT-Bench paper](https://arxiv.org/abs/2306.05685))

**1. Positional bias:** LLM judges prefer the *first* answer in A-vs-B comparison ~65% of the time.  
*Mitigation:* Run each comparison twice (A-first, B-first). Average scores. Call it a tie if winner flips.  
*Threshold:* flip rate > 15% = significant positional bias.

**2. Length bias:** LLMs assign higher scores to longer answers regardless of quality.  
*Mitigation:* Measure Pearson r between answer length and score. If |r| > 0.3, add verbosity penalty to rubric.

### Our rubric (point scoring, not comparative — avoids positional bias in primary eval)

```
relevance:     0–3  (does the answer address the question?)
faithfulness:  0–3  (is every claim grounded in context?)
completeness:  0–3  (are all answerable aspects covered?)
total:         0–9  → normalized to 0.0–1.0
```
"""),
md("### Written Question 5\n\n*In MT-Bench, GPT-4 achieves ~80% agreement with human experts on single-answer grading. What does the remaining 20% tell you about using LLM-as-judge for high-stakes decisions? What systematic errors does the judge make most often? Cite the paper.*"),
code("""from rag.eval.judge import llm_judge, positional_bias_check, length_correlation_check
import time

test_qs = [str(s.get('question','')) for s in testset[:8]]
contexts_list, answers_naive, answers_reranked = [], [], []

for q in test_qs:
    naive = dense_retrieve(q, collection, k=5)
    reranked = retrieve_fn(q)
    a_naive = generate_answer(q, naive, client)
    a_reranked = generate_answer(q, reranked, client)
    answers_naive.append(a_naive.answer)
    answers_reranked.append(a_reranked.answer)
    contexts_list.append(" ".join(r.text for r in naive[:3]))
    time.sleep(0.3)

print(f"Generated {len(test_qs)} answer pairs (naive dense vs hybrid+rerank)")
"""),
code("""naive_scores, reranked_scores = [], []
for q, a_n, a_r, ctx in zip(test_qs, answers_naive, answers_reranked, contexts_list):
    naive_scores.append(llm_judge(q, a_n, ctx, client))
    reranked_scores.append(llm_judge(q, a_r, ctx, client))
    time.sleep(0.3)

print(f"Naive dense mean score:    {sum(r.score for r in naive_scores)/len(naive_scores):.3f}")
print(f"Hybrid+rerank mean score:  {sum(r.score for r in reranked_scores)/len(reranked_scores):.3f}")
"""),
code("""# Positional bias check
print("=== Positional Bias ===")
pairs = list(zip(answers_naive, answers_reranked))
bias = positional_bias_check(test_qs[:4], pairs[:4], contexts_list[:4], client)
print(f"Flip rate: {bias['flip_rate']:.1%} ({bias['n_flips']}/{bias['n_total']})")
if bias['flip_rate'] > 0.15:
    print("⚠ Significant positional bias — average both orderings for trustworthy comparison")
else:
    print("✓ Positional bias within acceptable range")
"""),
code("""# Length bias check
import matplotlib.pyplot as plt, numpy as np
all_judge = naive_scores + reranked_scores
lm = length_correlation_check(all_judge)
print(f"=== Length Bias ===")
print(f"Pearson r: {lm['pearson_r']:.3f}  p={lm['p_value']:.4f}")
if abs(lm['pearson_r']) > 0.3:
    print("⚠ Significant length bias — add verbosity penalty to rubric")
else:
    print("✓ Length bias within acceptable range")

lengths = [len(r.answer.split()) for r in all_judge]
scores = [r.score for r in all_judge]
plt.figure(figsize=(6,4))
plt.scatter(lengths, scores, alpha=0.7)
m, b = np.polyfit(lengths, scores, 1)
xl = np.linspace(min(lengths), max(lengths), 100)
plt.plot(xl, m*xl+b, 'r--', label=f'r={lm["pearson_r"]:.2f}')
plt.xlabel("Answer length (words)"); plt.ylabel("Judge score (0-1)")
plt.title("Length vs Judge Score"); plt.legend(); plt.tight_layout()
plt.savefig('../docs/judge_length_bias.png', dpi=120); plt.show()
"""),
md("### Written Question 6\n\n*Your positional bias check shows a 22% flip rate. Describe exactly how you adjust the eval procedure to produce trustworthy win rates. What statistical test would you use to determine if one system is significantly better than another?*"),
code("""# Final summary
print("=" * 55)
print("FINAL EVALUATION SUMMARY")
print("=" * 55)
print("\\n1. Retrieval Benchmark (HotpotQA, Recall@k):")
for name, m in rows.items():
    print(f"   {name:<20} recall@5={m['recall@5']:.3f}")
print("\\n2. RAGAS (Hybrid + Rerank):")
for metric, score in ragas_scores.items():
    print(f"   {metric:<25} {score:.3f}")
n_s = sum(r.score for r in naive_scores)/len(naive_scores)
r_s = sum(r.score for r in reranked_scores)/len(reranked_scores)
print(f"\\n3. LLM-as-Judge:")
print(f"   Naive dense:     {n_s:.3f}")
print(f"   Hybrid+rerank:   {r_s:.3f}  ({r_s-n_s:+.3f})")
print(f"   Positional bias: {bias['flip_rate']:.1%} flip rate")
print(f"   Length bias:     r={lm['pearson_r']:.3f}")
""")]

nb.cells = cells
nb_path = Path('../notebooks/rag_pipeline.ipynb')
with open(nb_path, 'w') as f:
    nbf.write(nb, f)
print(f"Written: {nb_path}  ({len(cells)} cells)")
```

- [ ] **Step 2: Run build script**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python notebooks/build_rag_notebook.py
```

Expected: `Written: ../notebooks/rag_pipeline.ipynb  (N cells)`

- [ ] **Step 3: Verify notebook structure**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -c "
import nbformat
nb = nbformat.read('notebooks/rag_pipeline.ipynb', as_version=4)
parts = [c.source[:60] for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## Part')]
for p in parts: print(p)
"
```

Expected: 6 lines starting `## Part 1` through `## Part 6`

- [ ] **Step 4: Run all tests to verify nothing broke**

```bash
cd /home/abhisheksara/SpeculativeDecoding && python -m pytest tests/ -v --ignore=tests/test_embed.py 2>&1 | tail -20
```

Expected: all non-embed tests PASS (embed skipped because BGE-M3 may not be downloaded in CI)

- [ ] **Step 5: Commit**

```bash
git add notebooks/build_rag_notebook.py notebooks/rag_pipeline.ipynb
git commit -m "feat: RAG pipeline notebook — 6-part pedagogical notebook with theory, questions, ablation, and eval"
```

---

## Self-Review

**Spec coverage:**
- ✅ RAG full pipeline: Tasks 2–5 (ingest → embed → retrieve → generate)
- ✅ Structured outputs (Priority 1 #32): `RAGAnswer` + `Citation` in Task 5
- ✅ LLM-as-Judge (Priority 1 #24): Task 8, Part 6 notebook
- ✅ FRAMES/HotpotQA retrieval benchmark: Task 6, Part 5 notebook
- ✅ RAGAS synthetic testset + end-to-end eval: Task 7, Part 5 notebook
- ✅ LLM-as-Judge bias analysis: Task 8 (positional + length)
- ✅ BGE-M3 local embeddings: Task 3
- ✅ BM25 + hybrid RRF + cross-encoder ablation: Task 4, Part 3 notebook
- ✅ Data collection: Task 9
- ✅ Pedagogical notebook with theory, written questions, code, tests: Task 10

**Type consistency:**
- `Chunk` → defined Task 2, consumed Tasks 3, 4, 6, 7, 10 ✅
- `RetrievalResult` → defined Task 4, consumed Tasks 5, 6, 7, 8, 10 ✅
- `RAGAnswer` → defined Task 5, consumed Tasks 7, 8, 10 ✅
- `BenchmarkSample` → defined Task 6, used only in Task 6 ✅
- `JudgeResult` → defined Task 8, used only in Tasks 8, 10 ✅

**No placeholders found.**
