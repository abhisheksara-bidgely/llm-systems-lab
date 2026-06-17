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
        JudgeResult("q", "This is a very long detailed answer covering everything in detail.", 0.9, "r"),
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
