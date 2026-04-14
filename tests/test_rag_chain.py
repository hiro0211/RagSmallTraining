"""lib/rag_chain.py のテスト — 検索、Reranking、メタデータフィルタ."""

from unittest.mock import patch, MagicMock
import pytest


class TestSearchRelevantDocuments:
    def test_returns_sources(self):
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

            assert "sources" in result
            assert len(result["sources"]) == 1
            assert result["sources"][0].content == "RAGとは検索拡張生成のことです。"

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
            assert result["sources"] == []

    def test_default_match_count_is_10(self):
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
            assert params["match_count"] == 10


class TestHybridSearch:
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

            search_relevant_documents("テスト", use_hybrid=True)

            rpc_name = mock_admin.return_value.rpc.call_args[0][0]
            assert rpc_name == "match_documents_hybrid"

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


class TestMetadataFilter:
    def test_metadata_filter_calls_filtered_rpc(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents(
                "テスト",
                use_hybrid=True,
                metadata_filter={"source_type": "method"},
            )

            rpc_name = mock_admin.return_value.rpc.call_args[0][0]
            assert rpc_name == "match_documents_hybrid_filtered"
            params = mock_admin.return_value.rpc.call_args[0][1]
            assert params["metadata_filter"] == {"source_type": "method"}

    def test_no_filter_uses_standard_hybrid(self):
        with (
            patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed,
            patch("lib.rag_chain.get_supabase_admin") as mock_admin,
        ):
            MockEmbed.return_value.embed_query.return_value = [0.1] * 1536
            mock_rpc = MagicMock()
            mock_rpc.execute.return_value.data = []
            mock_admin.return_value.rpc.return_value = mock_rpc

            from lib.rag_chain import search_relevant_documents

            search_relevant_documents("テスト", metadata_filter=None)

            rpc_name = mock_admin.return_value.rpc.call_args[0][0]
            assert rpc_name == "match_documents_hybrid"


class TestRerankDocuments:
    def test_rerank_returns_top_k(self):
        from lib.rag_chain import Source, rerank_documents, _get_cross_encoder

        _get_cross_encoder.cache_clear()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]

        with patch.object(
            __import__("lib.rag_chain", fromlist=["_get_cross_encoder"]),
            "_get_cross_encoder",
            return_value=mock_model,
        ):
            sources = [
                Source(content=f"doc{i}", metadata={}, similarity=0.5)
                for i in range(5)
            ]

            result = rerank_documents("質問", sources, top_k=3)

            assert len(result) == 3
            assert result[0].similarity == 0.9
            assert result[1].similarity == 0.7
            assert result[2].similarity == 0.5

    def test_rerank_returns_all_if_fewer_than_top_k(self):
        from lib.rag_chain import Source, rerank_documents

        sources = [
            Source(content="doc1", metadata={}, similarity=0.5),
            Source(content="doc2", metadata={}, similarity=0.6),
        ]

        result = rerank_documents("質問", sources, top_k=5)

        assert len(result) == 2

    def test_rerank_returns_empty_for_empty_input(self):
        from lib.rag_chain import rerank_documents

        result = rerank_documents("質問", [], top_k=5)
        assert result == []


class TestFormatSourcesAsContext:
    def test_formats_with_source_labels(self):
        from lib.rag_chain import Source, format_sources_as_context

        sources = [
            Source(
                content="RAGの説明",
                metadata={"source": "test.md", "section": "RAG概要"},
                similarity=0.9,
            ),
            Source(
                content="ベクトル検索の説明",
                metadata={"source": "test.md", "section": "ベクトル検索"},
                similarity=0.8,
            ),
        ]

        context = format_sources_as_context(sources)

        assert "[出典1: test.md - RAG概要]" in context
        assert "[出典2: test.md - ベクトル検索]" in context
        assert "RAGの説明" in context

    def test_formats_without_section(self):
        from lib.rag_chain import Source, format_sources_as_context

        sources = [
            Source(content="内容", metadata={"source": "doc.txt"}, similarity=0.9),
        ]

        context = format_sources_as_context(sources)

        assert "[出典1: doc.txt]" in context


class TestBuildRagPrompt:
    def test_includes_context_and_question(self):
        from lib.rag_chain import build_rag_prompt

        messages = build_rag_prompt(question="RAGとは？", context="RAGの説明")
        assert len(messages) == 2
        assert "RAGの説明" in messages[0]["content"]
        assert messages[1]["content"] == "RAGとは？"


class TestEmbeddingsCache:
    def test_embeddings_client_cached(self):
        from lib.rag_chain import _get_embeddings

        _get_embeddings.cache_clear()

        with patch("lib.rag_chain.OpenAIEmbeddings") as MockEmbed:
            MockEmbed.return_value = MagicMock()
            result1 = _get_embeddings()
            result2 = _get_embeddings()
            assert result1 is result2
            MockEmbed.assert_called_once()
