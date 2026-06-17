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
