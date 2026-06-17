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
