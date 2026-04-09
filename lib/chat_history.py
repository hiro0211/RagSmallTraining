"""Chat session CRUD operations using Supabase."""

from datetime import datetime, timezone

from lib.supabase_client import get_supabase_admin


def create_session(title: str = "新しい会話") -> dict:
    """Create a new chat session and return it."""
    supabase = get_supabase_admin()
    result = supabase.table("chat_sessions").insert({"title": title}).execute()
    return result.data[0]


def list_sessions(limit: int = 10) -> list[dict]:
    """List recent chat sessions ordered by updated_at desc."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("chat_sessions")
        .select("id, title, updated_at")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def get_messages(session_id: str) -> list[dict]:
    """Get all messages for a session, ordered by created_at."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data


def save_message(session_id: str, role: str, content: str) -> None:
    """Save a message and update the session's updated_at timestamp."""
    supabase = get_supabase_admin()
    supabase.table("chat_messages").insert(
        {"session_id": session_id, "role": role, "content": content}
    ).execute()
    supabase.table("chat_sessions").update(
        {"updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def update_session_title(session_id: str, title: str) -> None:
    """Update a session's title."""
    supabase = get_supabase_admin()
    supabase.table("chat_sessions").update({"title": title}).eq(
        "id", session_id
    ).execute()
