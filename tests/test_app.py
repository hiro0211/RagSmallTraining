"""Tests for the Streamlit chat app logic."""

from unittest.mock import patch, MagicMock


class TestGenerateResponse:
    """Tests for generate_response() which delegates to LangGraph."""

    @patch("lib.chat.stream_response")
    def test_delegates_to_stream_response(self, mock_stream):
        from lib.chat import generate_response

        mock_stream.return_value = iter(["回答", "テスト"])

        result = list(generate_response("RAGとは？"))

        mock_stream.assert_called_once_with("RAGとは？", [])
        assert result == ["回答", "テスト"]

    @patch("lib.chat.stream_response")
    def test_passes_history_to_stream_response(self, mock_stream):
        from lib.chat import generate_response

        mock_stream.return_value = iter(["回答"])
        history = [
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の回答"},
        ]

        list(generate_response("新しい質問", history))

        mock_stream.assert_called_once_with("新しい質問", history)

    @patch("lib.chat.stream_response")
    def test_defaults_history_to_empty_list(self, mock_stream):
        from lib.chat import generate_response

        mock_stream.return_value = iter([])

        list(generate_response("質問"))

        mock_stream.assert_called_once_with("質問", [])

    @patch("lib.chat.stream_response")
    def test_yields_tokens_from_stream(self, mock_stream):
        from lib.chat import generate_response

        mock_stream.return_value = iter(["こん", "にち", "は"])

        tokens = list(generate_response("挨拶して"))

        assert tokens == ["こん", "にち", "は"]


class TestGenerateResponseWithSources:
    """Tests for generate_response_with_sources()."""

    @patch("lib.chat.stream_response_with_sources")
    def test_returns_token_generator_and_sources(self, mock_stream):
        from lib.chat import generate_response_with_sources
        from lib.rag_chain import Source

        sources = [Source(content="doc", metadata={}, similarity=0.9)]
        mock_stream.return_value = (iter(["回答"]), sources)

        token_gen, result_sources = generate_response_with_sources("質問")

        assert list(token_gen) == ["回答"]
        assert result_sources == sources
