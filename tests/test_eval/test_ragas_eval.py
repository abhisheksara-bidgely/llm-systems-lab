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
        result = run_ragas_eval(
            TESTSET,
            MagicMock(return_value=[]),
            MagicMock(return_value=MagicMock(answer="test")),
        )
    assert all(k in result for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"])
    assert all(0.0 <= v <= 1.0 for v in result.values())
