"""Tests for lib/graph.py - LangGraph RAG pipeline."""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestRAGState:
    """Tests for RAGState TypedDict."""

    def test_state_has_required_keys(self):
        from lib.graph import RAGState

        # RAGState should accept messages, context, and sources
        state: RAGState = {
            "messages": [HumanMessage(content="test")],
            "context": "some context",
            "sources": [],
        }
        assert "messages" in state
        assert "context" in state
        assert "sources" in state


class TestRetrieveNode:
    """Tests for the retrieve graph node."""

    @patch("lib.graph.search_relevant_documents")
    def test_extracts_question_from_last_message(self, mock_search):
        from lib.graph import retrieve

        mock_search.return_value = {"context": "doc content", "sources": []}

        state = {
            "messages": [
                HumanMessage(content="前の質問"),
                AIMessage(content="前の回答"),
                HumanMessage(content="最新の質問"),
            ],
            "context": "",
            "sources": [],
        }

        retrieve(state)

        mock_search.assert_called_once_with("最新の質問")

    @patch("lib.graph.search_relevant_documents")
    def test_returns_context_and_sources(self, mock_search):
        from lib.graph import retrieve
        from lib.rag_chain import Source

        sources = [Source(content="doc", metadata={}, similarity=0.9)]
        mock_search.return_value = {"context": "found context", "sources": sources}

        state = {
            "messages": [HumanMessage(content="質問")],
            "context": "",
            "sources": [],
        }

        result = retrieve(state)

        assert result["context"] == "found context"
        assert result["sources"] == sources

    @patch("lib.graph.search_relevant_documents")
    def test_handles_empty_search_results(self, mock_search):
        from lib.graph import retrieve

        mock_search.return_value = {"context": "", "sources": []}

        state = {
            "messages": [HumanMessage(content="存在しない質問")],
            "context": "",
            "sources": [],
        }

        result = retrieve(state)

        assert result["context"] == ""
        assert result["sources"] == []


class TestGenerateNode:
    """Tests for the generate graph node."""

    @patch("lib.graph.ChatOpenAI")
    def test_constructs_system_prompt_with_context(self, mock_chat_cls):
        from lib.graph import generate

        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="回答です")

        state = {
            "messages": [HumanMessage(content="質問")],
            "context": "テストコンテキスト",
            "sources": [],
        }

        generate(state)

        # Check the system message was included
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert isinstance(system_msg, SystemMessage)
        assert "テストコンテキスト" in system_msg.content

    @patch("lib.graph.ChatOpenAI")
    def test_trims_messages_to_last_20(self, mock_chat_cls):
        from lib.graph import generate

        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="回答")

        # Create 25 messages (more than 20 limit)
        messages = []
        for i in range(13):
            messages.append(HumanMessage(content=f"質問{i}"))
            if i < 12:
                messages.append(AIMessage(content=f"回答{i}"))

        state = {"messages": messages, "context": "ctx", "sources": []}

        generate(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # First message is SystemMessage, rest are trimmed conversation
        non_system = [m for m in call_args if not isinstance(m, SystemMessage)]
        assert len(non_system) <= 20

    @patch("lib.graph.ChatOpenAI")
    def test_uses_gpt4o_mini_with_temperature_0(self, mock_chat_cls):
        from lib.graph import generate

        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="回答")

        state = {
            "messages": [HumanMessage(content="質問")],
            "context": "ctx",
            "sources": [],
        }

        generate(state)

        mock_chat_cls.assert_called_with(
            model="gpt-4o-mini", temperature=0, streaming=True
        )

    @patch("lib.graph.ChatOpenAI")
    def test_returns_ai_message(self, mock_chat_cls):
        from lib.graph import generate

        mock_llm = MagicMock()
        mock_chat_cls.return_value = mock_llm
        ai_msg = AIMessage(content="これは回答です")
        mock_llm.invoke.return_value = ai_msg

        state = {
            "messages": [HumanMessage(content="質問")],
            "context": "ctx",
            "sources": [],
        }

        result = generate(state)

        assert result["messages"] == [ai_msg]


class TestBuildRagGraph:
    """Tests for build_rag_graph()."""

    def test_compiles_without_error(self):
        from lib.graph import build_rag_graph

        graph = build_rag_graph()
        assert graph is not None

    def test_graph_has_retrieve_and_generate_nodes(self):
        from lib.graph import build_rag_graph

        graph = build_rag_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "retrieve" in node_names
        assert "generate" in node_names


class TestStreamResponse:
    """Tests for stream_response()."""

    @patch("lib.graph.build_rag_graph")
    def test_yields_string_tokens(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph

        # Simulate stream_mode="messages" v2 output
        mock_graph.stream.return_value = [
            {
                "type": "messages",
                "data": (
                    MagicMock(content="こん"),
                    {"langgraph_node": "generate"},
                ),
            },
            {
                "type": "messages",
                "data": (
                    MagicMock(content="にちは"),
                    {"langgraph_node": "generate"},
                ),
            },
        ]

        tokens = list(stream_response("質問", []))

        assert tokens == ["こん", "にちは"]

    @patch("lib.graph.build_rag_graph")
    def test_converts_history_dicts_to_langchain_messages(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = []

        history = [
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の回答"},
        ]

        list(stream_response("新しい質問", history))

        call_args = mock_graph.stream.call_args[0][0]
        messages = call_args["messages"]
        assert len(messages) == 3  # 2 history + 1 current
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "新しい質問"

    @patch("lib.graph.build_rag_graph")
    def test_filters_tokens_to_generate_node_only(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = [
            {
                "type": "messages",
                "data": (
                    MagicMock(content="retrieve output"),
                    {"langgraph_node": "retrieve"},
                ),
            },
            {
                "type": "messages",
                "data": (
                    MagicMock(content="generate output"),
                    {"langgraph_node": "generate"},
                ),
            },
            {
                "type": "updates",
                "data": {"some": "update"},
            },
        ]

        tokens = list(stream_response("質問", []))

        assert tokens == ["generate output"]

    @patch("lib.graph.build_rag_graph")
    def test_handles_empty_history(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = []

        list(stream_response("質問", []))

        call_args = mock_graph.stream.call_args[0][0]
        messages = call_args["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)

    @patch("lib.graph.stream_response")
    @patch("lib.graph.search_relevant_documents")
    def test_stream_response_with_sources_returns_sources(
        self, mock_search, mock_stream
    ):
        from lib.graph import stream_response_with_sources
        from lib.rag_chain import Source

        sources = [Source(content="doc", metadata={"source": "test.md"}, similarity=0.9)]
        mock_search.return_value = {"context": "ctx", "sources": sources}
        mock_stream.return_value = iter(["回答"])

        token_gen, result_sources = stream_response_with_sources("質問", [])
        tokens = list(token_gen)

        assert tokens == ["回答"]
        assert len(result_sources) == 1
        assert result_sources[0].similarity == 0.9
        mock_search.assert_called_once_with("質問")
        mock_stream.assert_called_once_with("質問", [])

    @patch("lib.graph.build_rag_graph")
    def test_skips_empty_content_chunks(self, mock_build):
        from lib.graph import stream_response

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.stream.return_value = [
            {
                "type": "messages",
                "data": (
                    MagicMock(content=""),
                    {"langgraph_node": "generate"},
                ),
            },
            {
                "type": "messages",
                "data": (
                    MagicMock(content="有効なトークン"),
                    {"langgraph_node": "generate"},
                ),
            },
        ]

        tokens = list(stream_response("質問", []))

        assert tokens == ["有効なトークン"]
