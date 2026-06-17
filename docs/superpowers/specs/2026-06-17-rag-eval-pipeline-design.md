# RAG + Eval Pipeline — Design Spec
Date: 2026-06-17

## Goal
Build a production/research-grade RAG system over AI research papers (arXiv) + engineering blogs (Anthropic, OpenAI, HuggingFace, Uber, Netflix, Grab, Razorpay). After completing this notebook, the learner should be able to:
1. Build custom end-to-end RAG pipelines independently at a startup
2. Critically read and evaluate RAG research papers (understand what claims are valid, what evals are rigorous)
3. Contribute to RAG research — know what open problems exist and how to measure progress

## Problem
Bridge the gap between AI research and production — answer questions like "What paper underpins how Anthropic handles context compression, and how do they apply it in their products?" Requires reasoning across two corpora (papers + blogs), making retrieval genuinely hard and eval genuinely necessary.

## Learner Profile
- DS background, 4 YOE, near-zero LLM eval knowledge
- Has OpenAI API access
- Wants production + research-level understanding, not surface familiarity
- Existing repo has a speculative decoding notebook as reference for depth and style

## Stack
- **Embeddings**: BGE-M3 (local, via `FlagEmbedding`) — no API cost per chunk, production-realistic
- **Vector DB**: ChromaDB (local, no infra setup)
- **Sparse retrieval**: BM25 via `rank_bm25`
- **Reranker**: BGE-reranker-v2-m3 (local cross-encoder)
- **Generation + Judge**: OpenAI GPT-4o-mini (structured outputs via Pydantic)
- **Eval**: RAGAS + FRAMES/HotpotQA subset

---

## Architecture

```
Data Sources
  arXiv papers (~50 PDFs)  +  Engineering blogs (~30 HTML/markdown)
          ↓
  Ingestion & Chunking
  semantic chunking (sentences → chunks ≤512 tokens)
  metadata: source_type, company/institution, topic_tag, pub_date
          ↓
  Embedding  →  ChromaDB  (BGE-M3, local)
          ↓
  Retrieval Layer
  Stage 1: BM25 + dense hybrid retrieval (RRF fusion)
  Stage 2: cross-encoder reranking (BGE-reranker)
          ↓
  Answer Generation
  GPT-4o-mini, structured output:
    { answer: str, citations: list[Citation], confidence: float }
          ↓
  Eval Harness
  A: Retrieval benchmark — FRAMES or HotpotQA subset (comparable to literature)
  B: End-to-end eval — RAGAS synthetic QA on actual corpus
     metrics: retrieval_recall, context_precision, faithfulness, answer_relevance
  C: LLM-as-Judge — rubric scoring + positional bias check + length-correlation check
  D: Ablation table — naive vs hybrid vs reranked, numbers at each stage
```

---

## Data

### Papers (~50 arXiv PDFs)
Topics: RAG, agents, fine-tuning (SFT/DPO), evals, inference optimization.
Key papers to include:
- RAG (Lewis et al. 2020), HyDE, RAPTOR, SELF-RAG
- ReAct, Toolformer, AgentBench
- InstructGPT (SFT), DPO paper, RLHF survey
- RAGAS paper, FRAMES paper, ARES paper
- Speculative decoding (already in repo)

### Blogs (~30 posts)
Sources: Anthropic engineering blog, OpenAI research blog, HuggingFace blog, Uber engineering, Netflix tech blog, Grab tech blog, Razorpay tech blog.
Selection criteria: posts that reference a specific paper OR describe a production implementation decision.

### Metadata per chunk
```python
{
  "source_type": "paper" | "blog",
  "company": str,           # "Anthropic", "Uber", etc.
  "topic_tag": str,         # "RAG", "agents", "fine-tuning", "eval", "inference"
  "pub_date": str,          # ISO format
  "url": str,
  "chunk_id": str
}
```

---

## Eval Design

### A — Retrieval Benchmark (FRAMES or HotpotQA)
- Use a 200-question subset from FRAMES (Google DeepMind, 2024) or HotpotQA
- **Why FRAMES**: unlike BEIR (which benchmarks retrieval only), FRAMES tests the full pipeline — factuality, retrieval, and multi-step reasoning. Published by Google DeepMind. Paper: https://arxiv.org/abs/2409.12941
- **Why HotpotQA**: gold standard multi-hop dataset, widely cited, enables direct comparison with published RAG papers. Paper: https://arxiv.org/abs/1809.09600
- Metric: Retrieval Recall@k (k=3, 5, 10)
- Note: these benchmarks use their own corpora — this measures retrieval component quality in a comparable, literature-standard way

### B — RAGAS End-to-End Eval
- Use RAGAS `TestsetGenerator` to generate 100-150 QA pairs from our actual corpus
- Question types: simple (40%), reasoning (30%), multi-hop (30%)
- Spot-check 25 manually — notebook section "Can we trust our synthetic test set?" with annotation results
- **Why RAGAS**: paper https://arxiv.org/abs/2309.15217. Automated metrics using LLM-as-evaluator, no human labels needed. Widely adopted in production.
- Metrics:
  - `retrieval_recall`: were the relevant chunks retrieved?
  - `context_precision`: were retrieved chunks actually useful?
  - `faithfulness`: is the answer grounded in retrieved context?
  - `answer_relevance`: does the answer address the question?

### C — LLM-as-Judge Bias Analysis
- Judge: GPT-4o-mini with rubric (0-3 scale: relevance, faithfulness, completeness)
- Positional bias check: run eval twice with A/B answer order swapped, measure % where verdict flips
- Length-correlation check: Pearson correlation between answer length and judge score
- If bias is significant (>15% flip rate or r>0.3), apply correction (average both orderings)
- **Why this matters**: LLM judges prefer longer answers and tend to favor the first option in comparisons. Unchecked, this invalidates the eval. Source: https://arxiv.org/abs/2306.05685

### D — Ablation Table
```
Strategy          | Recall@5 | Context Precision | Faithfulness | Latency
------------------|----------|-------------------|--------------|--------
Naive dense       |          |                   |              |
BM25 only         |          |                   |              |
Hybrid (RRF)      |          |                   |              |
Hybrid + rerank   |          |                   |              |
```
Numbers filled in during notebook execution. This is the portfolio centerpiece.

---

## Structured Output Schema

```python
from pydantic import BaseModel

class Citation(BaseModel):
    source_title: str
    source_type: str   # "paper" | "blog"
    company: str
    url: str
    relevant_excerpt: str

class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float          # 0.0–1.0
    reasoning_steps: list[str] # for multi-hop questions
```

---

## Notebook Structure (`notebooks/rag_pipeline.ipynb`)

### Part 1 — Data Ingestion & Chunking
- Theory: why chunking strategy determines retrieval quality (chunk too large → low precision, too small → low recall)
- Semantic chunking: split on sentence boundaries, merge until ≤512 tokens
- Metadata tagging pipeline
- Code blank: `chunk_document(text, metadata) -> list[Chunk]`
- Test: chunk count, mean chunk length, no chunk > 512 tokens

### Part 2 — Embedding & Vector Store
- Theory: dense vs sparse retrieval — what each captures, why neither alone is sufficient
- BGE-M3 explanation + link (https://arxiv.org/abs/2402.03216): why it beats OpenAI embeddings on multilingual and technical text at zero cost
- Code blank: `embed_and_store(chunks, collection_name) -> ChromaCollection`
- Test: embedding shape, retrieval sanity check on 5 known queries

### Part 3 — Retrieval Ablation
- Theory: BM25 (keyword match) + dense (semantic match) → why hybrid via RRF beats both
- Cross-encoder reranking: why bi-encoder retrieval ≠ bi-encoder ranking
- BGE-reranker link: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Code blanks: `bm25_retrieve`, `dense_retrieve`, `hybrid_rrf`, `rerank`
- Benchmark: run all 4 strategies on FRAMES/HotpotQA subset, fill ablation table

### Part 4 — Structured Answer Generation
- Theory: why free-text generation fails at scale (unparseable, inconsistent citations)
- OpenAI structured outputs with Pydantic — `instructor` library
- Code blank: `generate_answer(query, chunks) -> RAGAnswer`
- Test: 20 queries, assert all responses parse to `RAGAnswer`, citations contain valid URLs

### Part 5 — RAGAS End-to-End Eval
- Theory: what RAGAS measures and why (link to paper)
- Synthetic testset generation: run `TestsetGenerator`, show question type distribution
- "Can we trust our test set?" — manual spot-check of 25 questions, annotation table
- Code: run full RAGAS eval across all 4 retrieval strategies
- Output: scores table per strategy, interpretation

### Part 6 — LLM-as-Judge Bias Analysis
- Theory: what LLM-as-judge is, why it's the primary eval method at startups (no human labels needed)
- Known biases: length preference, positional bias (link: https://arxiv.org/abs/2306.05685)
- "Can we trust our judge?" — positional bias experiment, length correlation plot
- Apply correction if needed
- Final trustworthy judge scores

---

## Module Structure (`src/rag/`)

```
src/rag/
  __init__.py
  ingest.py       # chunk_document, load_pdf, load_html
  embed.py        # embed_and_store, load_collection
  retrieve.py     # bm25_retrieve, dense_retrieve, hybrid_rrf, rerank
  generate.py     # generate_answer, RAGAnswer schema
  eval/
    __init__.py
    ragas_eval.py       # run_ragas_eval, generate_testset
    retrieval_bench.py  # run_frames_eval, run_hotpotqa_eval
    judge.py            # llm_judge, positional_bias_check, length_correlation
```

---

## Directory Structure

```
SpeculativeDecoding/
  data/
    papers/           # arXiv PDFs
    blogs/            # scraped HTML/markdown
    eval/
      frames_subset.json
      hotpotqa_subset.json
      synthetic_testset.json
  src/
    rag/              # module (see above)
  notebooks/
    rag_pipeline.ipynb
  docs/
    superpowers/
      specs/
        2026-06-17-rag-eval-pipeline-design.md
```

---

## Dependencies to Add

```
# Retrieval
rank-bm25
FlagEmbedding          # BGE-M3 + reranker
chromadb

# Generation
instructor             # structured outputs with OpenAI
openai

# Eval
ragas
datasets               # for FRAMES/HotpotQA

# Ingestion
pypdf
beautifulsoup4
```

---

## What This Covers (Career Guide Blockers)
- **RAG** (Priority 1 #20): full pipeline, chunking → embedding → retrieval → reranking → eval
- **Structured Outputs** (Priority 1 #32): Pydantic schema, instructor, citations
- **LLM-as-Judge** (Priority 1 #24): RAGAS eval + bias analysis section
- **Benchmark Contamination awareness** (Priority 2 #23): FRAMES vs BEIR discussion
- **Chatbot Arena / benchmarks** (Priority 2 #25): FRAMES, HotpotQA discussion in notebook

## What This Does NOT Cover (separate projects)
- Tool Use / Function Calling
- ReAct / Agents
- SFT / Fine-tuning
- DPO
