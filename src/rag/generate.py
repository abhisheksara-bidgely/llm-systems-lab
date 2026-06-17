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
