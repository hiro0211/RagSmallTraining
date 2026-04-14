"""lib/graph.py のテスト — LangGraph RAGパイプライン."""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestRAGState:
    def test_state_has_required_keys(self):
        from lib.graph import RAGState

        state: RAGState = {
            "messages": [HumanMessage(content="test")],
            "rewritten_query": "",
            "hyde_context": "",
            "expanded_queries": [],
            "context": "",
            "sources": [],
            "hyde_sources": [],
            "multi_query_sources": [],
            "original_sources": [],
            "model_id": "gpt-4o-mini",
        }
        assert "hyde_context" in state
        assert "expanded_queries" in state


class TestRewriteQueryNode:
    @patch("lib.graph.create_llm")
    def test_rewrites_abbreviation(self, mock_create_llm):
        from lib.graph import rewrite_query

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="RAG とは何ですか")

        state = {
            "messages": [HumanMessage(content="ラグって何")],
            "rewritten_query": "",
            "model_id": "",
        }

        result = rewrite_query(state)
        assert "RAG" in result["rewritten_query"]

    @patch("lib.graph.create_llm")
    def test_falls_back_on_empty_response(self, mock_create_llm):
        from lib.graph import rewrite_query

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="   ")

        state = {
            "messages": [HumanMessage(content="元の質問")],
            "rewritten_query": "",
            "model_id": "",
        }

        result = rewrite_query(state)
        assert result["rewritten_query"] == "元の質問"


class TestHydeQueryNode:
    @patch("lib.graph.search_relevant_documents")
    @patch("lib.graph.create_llm")
    def test_generates_hypothesis_and_searches(self, mock_create_llm, mock_search):
        from lib.graph import hyde_query
        from lib.rag_chain import Source

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="仮説回答テキスト")

        mock_search.return_value = {
            "sources": [Source(content="doc", metadata={}, similarity=0.8)],
        }

        state = {
            "messages": [HumanMessage(content="質問")],
            "rewritten_query": "リライト済み質問",
            "model_id": "",
        }

        result = hyde_query(state)

        assert result["hyde_context"] == "仮説回答テキスト"
        assert len(result["hyde_sources"]) == 1
        # 仮説回答で検索している
        mock_search.assert_called_once_with("仮説回答テキスト", match_count=10)


class TestMultiQueryExpandNode:
    @patch("lib.graph.search_relevant_documents")
    @patch("lib.graph.create_llm")
    def test_generates_expanded_queries(self, mock_create_llm, mock_search):
        from lib.graph import multi_query_expand

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content="言い換え1\n言い換え2\n言い換え3"
        )

        mock_search.return_value = {"sources": []}

        state = {
            "messages": [HumanMessage(content="質問")],
            "rewritten_query": "リライト済み",
            "model_id": "",
        }

        result = multi_query_expand(state)

        assert len(result["expanded_queries"]) == 3
        assert mock_search.call_count == 3

    @patch("lib.graph.search_relevant_documents")
    @patch("lib.graph.create_llm")
    def test_limits_to_3_queries(self, mock_create_llm, mock_search):
        from lib.graph import multi_query_expand

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content="q1\nq2\nq3\nq4\nq5"
        )

        mock_search.return_value = {"sources": []}

        state = {
            "messages": [HumanMessage(content="質問")],
            "rewritten_query": "リライト済み",
            "model_id": "",
        }

        result = multi_query_expand(state)

        assert len(result["expanded_queries"]) <= 3


class TestMergeResultsNode:
    def test_merges_and_deduplicates(self):
        from lib.graph import merge_results
        from lib.rag_chain import Source

        state = {
            "hyde_sources": [
                Source(content="同じドキュメント", metadata={}, similarity=0.9),
            ],
            "multi_query_sources": [
                Source(content="同じドキュメント", metadata={}, similarity=0.8),
                Source(content="別のドキュメント", metadata={}, similarity=0.7),
            ],
            "original_sources": [
                Source(content="3つ目のドキュメント", metadata={}, similarity=0.6),
            ],
        }

        result = merge_results(state)

        # 重複除去で3件になる
        assert len(result["sources"]) == 3

    def test_handles_empty_sources(self):
        from lib.graph import merge_results

        state = {
            "hyde_sources": [],
            "multi_query_sources": [],
            "original_sources": [],
        }

        result = merge_results(state)
        assert result["sources"] == []


class TestRerankNode:
    @patch("lib.graph.rerank_documents")
    @patch("lib.graph.format_sources_as_context")
    def test_reranks_and_formats_context(self, mock_format, mock_rerank):
        from lib.graph import rerank
        from lib.rag_chain import Source

        reranked = [Source(content="top", metadata={}, similarity=0.95)]
        mock_rerank.return_value = reranked
        mock_format.return_value = "[出典1] top"

        state = {
            "messages": [HumanMessage(content="質問")],
            "rewritten_query": "リライト済み",
            "sources": [
                Source(content="top", metadata={}, similarity=0.9),
                Source(content="low", metadata={}, similarity=0.3),
            ],
        }

        result = rerank(state)

        assert result["sources"] == reranked
        assert result["context"] == "[出典1] top"
        mock_rerank.assert_called_once()


class TestGenerateNode:
    @patch("lib.graph.create_llm")
    def test_includes_context_in_system_prompt(self, mock_create_llm):
        from lib.graph import generate

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="回答")

        state = {
            "messages": [HumanMessage(content="質問")],
            "context": "テストコンテキスト",
            "sources": [],
            "model_id": "",
        }

        generate(state)

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert isinstance(system_msg, SystemMessage)
        assert "テストコンテキスト" in system_msg.content


class TestBuildRagGraph:
    def test_compiles_without_error(self):
        from lib.graph import build_rag_graph

        graph = build_rag_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        from lib.graph import build_rag_graph

        graph = build_rag_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "rewrite_query" in node_names
        assert "hyde_query" in node_names
        assert "multi_query_expand" in node_names
        assert "retrieve" in node_names
        assert "merge_results" in node_names
        assert "rerank" in node_names
        assert "generate" in node_names


class TestGetCompiledGraph:
    def test_returns_same_instance(self):
        from lib.graph import get_compiled_graph

        get_compiled_graph.cache_clear()
        g1 = get_compiled_graph()
        g2 = get_compiled_graph()
        assert g1 is g2


class TestStreamResponse:
    @patch("lib.graph.build_rag_graph")
    def test_yields_generate_tokens_only(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = [
            {"type": "messages", "data": (MagicMock(content="retrieve出力"), {"langgraph_node": "retrieve"})},
            {"type": "messages", "data": (MagicMock(content="回答"), {"langgraph_node": "generate"})},
        ]

        tokens = list(stream_response("質問", []))
        assert tokens == ["回答"]

    @patch("lib.graph.build_rag_graph")
    def test_passes_model_id(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = []

        list(stream_response("質問", [], model_id="gemini-2.5-flash"))

        call_args = mock_graph.stream.call_args[0][0]
        assert call_args["model_id"] == "gemini-2.5-flash"

    @patch("lib.graph.get_compiled_graph")
    def test_stream_with_sources_returns_rerank_sources(self, mock_get_graph):
        from lib.graph import stream_response_with_sources
        from lib.rag_chain import Source

        sources = [Source(content="doc", metadata={}, similarity=0.9)]
        mock_graph = MagicMock()
        mock_get_graph.return_value = mock_graph
        mock_graph.stream.return_value = [
            {"type": "updates", "data": {"rerank": {"sources": sources, "context": "ctx"}}},
            {"type": "messages", "data": (MagicMock(content="回答"), {"langgraph_node": "generate"})},
        ]

        token_gen, result_sources = stream_response_with_sources("質問", [])
        tokens = list(token_gen)

        assert tokens == ["回答"]
        assert len(result_sources) == 1
        assert result_sources[0].similarity == 0.9
