import pytest
from src.analysis.vector_store import create_vector_store, chunk_text

def test_chunk_text():
    text = "This is a test " * 100  # Create long text
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_create_vector_store(tmp_path):
    chunks = ["Test chunk 1", "Test chunk 2"]
    store = create_vector_store(chunks, str(tmp_path / "test_vectors"))
    assert store is not None
    
    # Test similarity search
    results = store.similarity_search("test", k=1)
    assert len(results) == 1