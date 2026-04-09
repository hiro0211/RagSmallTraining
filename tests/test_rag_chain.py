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
    def test_default_threshold_is_0_5(self):
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

            call_args = mock_admin.return_value.rpc.call_args[0]
            params = mock_admin.return_value.rpc.call_args[0][1]
            assert params["match_threshold"] == 0.5


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

    def test_system_prompt_includes_fallback_instruction(self):
        from lib.rag_chain import build_rag_prompt

        messages = build_rag_prompt(question="test", context="some context")
        assert "ナレッジベースに含まれていません" in messages[0]["content"]


class TestRagSystemPrompt:
    def test_prompt_template_exists(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "コンテキスト" in RAG_SYSTEM_PROMPT
        assert "{context}" in RAG_SYSTEM_PROMPT

    def test_prompt_includes_citation_instruction(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "引用" in RAG_SYSTEM_PROMPT

    def test_prompt_prohibits_speculation(self):
        from lib.rag_chain import RAG_SYSTEM_PROMPT

        assert "推測" in RAG_SYSTEM_PROMPT


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
