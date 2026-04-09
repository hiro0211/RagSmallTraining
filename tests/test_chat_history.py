"""Tests for lib/chat_history.py - Chat session CRUD operations."""

from unittest.mock import patch, MagicMock


class TestCreateSession:
    """Tests for create_session()."""

    @patch("lib.chat_history.get_supabase_admin")
    def test_creates_session_with_default_title(self, mock_get_admin):
        from lib.chat_history import create_session

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "abc-123", "title": "新しい会話", "created_at": "2026-04-09T00:00:00Z"}
        ]

        result = create_session()

        mock_client.table.assert_called_with("chat_sessions")
        mock_client.table().insert.assert_called_with({"title": "新しい会話"})
        assert result["id"] == "abc-123"
        assert result["title"] == "新しい会話"

    @patch("lib.chat_history.get_supabase_admin")
    def test_creates_session_with_custom_title(self, mock_get_admin):
        from lib.chat_history import create_session

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "abc-456", "title": "RAGについて", "created_at": "2026-04-09T00:00:00Z"}
        ]

        result = create_session(title="RAGについて")

        mock_client.table().insert.assert_called_with({"title": "RAGについて"})
        assert result["title"] == "RAGについて"


class TestListSessions:
    """Tests for list_sessions()."""

    @patch("lib.chat_history.get_supabase_admin")
    def test_returns_sessions_ordered_by_updated_at(self, mock_get_admin):
        from lib.chat_history import list_sessions

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        sessions_data = [
            {"id": "s1", "title": "最新", "updated_at": "2026-04-09T02:00:00Z"},
            {"id": "s2", "title": "古い", "updated_at": "2026-04-09T01:00:00Z"},
        ]
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = sessions_data

        result = list_sessions(limit=10)

        mock_client.table.assert_called_with("chat_sessions")
        assert len(result) == 2
        assert result[0]["title"] == "最新"

    @patch("lib.chat_history.get_supabase_admin")
    def test_limits_to_specified_count(self, mock_get_admin):
        from lib.chat_history import list_sessions

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []

        list_sessions(limit=5)

        mock_client.table().select().order().limit.assert_called_with(5)


class TestGetMessages:
    """Tests for get_messages()."""

    @patch("lib.chat_history.get_supabase_admin")
    def test_returns_messages_for_session(self, mock_get_admin):
        from lib.chat_history import get_messages

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        messages_data = [
            {"role": "user", "content": "RAGとは？"},
            {"role": "assistant", "content": "RAGは..."},
        ]
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = messages_data

        result = get_messages("session-1")

        mock_client.table.assert_called_with("chat_messages")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "RAGとは？"}

    @patch("lib.chat_history.get_supabase_admin")
    def test_returns_empty_list_for_no_messages(self, mock_get_admin):
        from lib.chat_history import get_messages

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = []

        result = get_messages("empty-session")

        assert result == []


class TestSaveMessage:
    """Tests for save_message()."""

    @patch("lib.chat_history.get_supabase_admin")
    def test_inserts_message_into_chat_messages(self, mock_get_admin):
        from lib.chat_history import save_message

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        save_message("session-1", "user", "RAGとは？")

        calls = mock_client.table.call_args_list
        # First call: insert into chat_messages
        assert calls[0].args[0] == "chat_messages"
        mock_client.table("chat_messages").insert.assert_called_with(
            {"session_id": "session-1", "role": "user", "content": "RAGとは？"}
        )

    @patch("lib.chat_history.get_supabase_admin")
    def test_updates_session_updated_at(self, mock_get_admin):
        from lib.chat_history import save_message

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        save_message("session-1", "user", "質問")

        # Second call: update chat_sessions
        mock_client.table("chat_sessions").update.assert_called_once()
        mock_client.table("chat_sessions").update().eq.assert_called_with("id", "session-1")


class TestUpdateSessionTitle:
    """Tests for update_session_title()."""

    @patch("lib.chat_history.get_supabase_admin")
    def test_updates_title(self, mock_get_admin):
        from lib.chat_history import update_session_title

        mock_client = MagicMock()
        mock_get_admin.return_value = mock_client
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        update_session_title("session-1", "RAGについて質問")

        mock_client.table.assert_called_with("chat_sessions")
        mock_client.table().update.assert_called_with({"title": "RAGについて質問"})
        mock_client.table().update().eq.assert_called_with("id", "session-1")
