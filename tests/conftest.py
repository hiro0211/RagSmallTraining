import os
import pytest


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret-key")


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear all lru_cache instances between tests."""
    yield
    from lib.llm import create_llm
    from lib.graph import get_compiled_graph
    from lib.rag_chain import _get_embeddings, _get_cross_encoder

    create_llm.cache_clear()
    get_compiled_graph.cache_clear()
    _get_embeddings.cache_clear()
    _get_cross_encoder.cache_clear()
