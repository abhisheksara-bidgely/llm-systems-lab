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
