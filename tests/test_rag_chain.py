from unittest.mock import patch, MagicMock
import pytest


class TestSearchRelevantDocuments:
    def test_returns_context_and_sources(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = [
                {
                    "id": 1,
                    "content": "RAGとは検索拡張生成のことです。",
                    "metadata": {"source": "test.md"},
                    "similarity": 0.9,
                },
            ]
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            result = search_relevant_documents("RAGとは？")

            assert "context" in result
            assert "sources" in result
            assert "RAGとは検索拡張生成のことです。" in result["context"]
            assert len(result["sources"]) == 1

    def test_returns_empty_when_no_matches(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            result = search_relevant_documents("今日の天気は？")
            assert result["context"] == ""
            assert result["sources"] == []


class TestDefaultThreshold:
    def test_default_threshold_is_0_3(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents("テスト")

            params = mock_admin.return_value.rpc.call_args[0][1]
            assert params["match_threshold"] == 0.3


class TestBuildRagPrompt:
    def test_includes_context_and_question(self):
        from lib.rag_chain import build_rag_prompt

        messages = build_rag_prompt(
            question="RAGとは？",
            context="RAGとは検索拡張生成のことです。",
        )
        assert len(messages) == 2
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "RAGとは検索拡張生成のことです。" in system_msg["content"]
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert "RAGとは？" in user_msg["content"]

    def test_system_prompt_includes_knowledge_source_markers(self):
        from lib.rag_chain import build_rag_prompt

        messages = build_rag_prompt(question="test", context="some context")
        system_content = messages[0]["content"]
        assert "【ナレッジベース】" in system_content
        assert "【一般知識】" in system_content


class TestRagSystemPrompt:
    def test_prompt_template_exists(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "コンテキスト" in RAG_SYSTEM_PROMPT
        assert "{context}" in RAG_SYSTEM_PROMPT

    def test_prompt_includes_citation_instruction(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "引用" in RAG_SYSTEM_PROMPT

    def test_prompt_allows_flexible_interpretation(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "表記ゆれ" in RAG_SYSTEM_PROMPT or "柔軟" in RAG_SYSTEM_PROMPT

    def test_prompt_includes_priority_based_answering(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "回答の優先順位" in RAG_SYSTEM_PROMPT

    def test_prompt_supports_general_knowledge_fallback(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "一般知識" in RAG_SYSTEM_PROMPT

    def test_prompt_does_not_reject_unknown_questions(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "この情報はナレッジベースに含まれていません" not in RAG_SYSTEM_PROMPT


class TestContextSourceLabels:
    def test_context_includes_source_labels(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = [
                {
                    "id": 1,
                    "content": "RAGとは検索拡張生成のことです。",
                    "metadata": {"source": "test.md", "section": "RAGの技術概要"},
                    "similarity": 0.9,
                },
                {
                    "id": 2,
                    "content": "ベクトル検索はコサイン類似度を使います。",
                    "metadata": {"source": "test.md", "section": "ベクトル検索"},
                    "similarity": 0.8,
                },
            ]
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            result = search_relevant_documents("RAGとは？")

            assert "[出典1:" in result["context"]
            assert "[出典2:" in result["context"]


class TestHybridSearch:
    """Tests for hybrid search (use_hybrid parameter)."""

    def test_hybrid_search_calls_hybrid_rpc(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents("Supabaseとは", use_hybrid=True)

            rpc_name = mock_admin.return_value.rpc.call_args[0][0]
            assert rpc_name == "match_documents_hybrid"

    def test_hybrid_search_passes_query_text(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents("pgvectorの使い方", use_hybrid=True)

            params = mock_admin.return_value.rpc.call_args[0][1]
            assert params["query_text"] == "pgvectorの使い方"

    def test_non_hybrid_search_calls_original_rpc(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents("テスト", use_hybrid=False)

            rpc_name = mock_admin.return_value.rpc.call_args[0][0]
            assert rpc_name == "match_documents"

    def test_hybrid_search_returns_combined_score(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = [
                {
                    "id": 1,
                    "content": "ハイブリッド検索の結果",
                    "metadata": {"source": "test.md"},
                    "similarity": 0.85,
                },
            ]
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            result = search_relevant_documents("Supabase", use_hybrid=True)

            assert len(result["sources"]) == 1
            assert result["sources"][0].similarity == 0.85


class TestEmbeddingsCache:
    """Tests for _get_embeddings() caching."""

    def test_embeddings_client_cached(self):
        from lib.rag_chain import _get_embeddings

        _get_embeddings.cache_clear()

        with patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed:
            MockEmbed.return_value = MagicMock()

            result1 = _get_embeddings()
            result2 = _get_embeddings()

            assert result1 is result2
            MockEmbed.assert_called_once()
