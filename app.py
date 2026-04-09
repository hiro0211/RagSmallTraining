"""RAG Small - Streamlit Chat UI with session management."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env.local")

from lib.chat import generate_response_with_sources
from lib.chat_history import (
    create_session,
    list_sessions,
    get_messages,
    save_message,
    update_session_title,
)

st.set_page_config(
    page_title="RAG Small",
    page_icon="🔍",
    layout="centered",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# --- Sidebar: Session Management ---
with st.sidebar:
    st.title("会話履歴")

    if st.button("＋ 新しい会話", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    sessions = list_sessions(limit=10)
    for session in sessions:
        is_active = st.session_state.current_session_id == session["id"]
        label = f"{'▶ ' if is_active else ''}{session['title']}"
        if st.button(label, key=session["id"], use_container_width=True):
            st.session_state.current_session_id = session["id"]
            st.session_state.messages = get_messages(session["id"])
            st.rerun()

# --- Main: Chat Area ---
st.title("RAG Small")
st.caption("社内ナレッジに基づいて質問に回答するアシスタント")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("質問を入力してください"):
    # Auto-create session if none exists
    if not st.session_state.current_session_id:
        session = create_session(title=prompt[:30])
        st.session_state.current_session_id = session["id"]

    session_id = st.session_state.current_session_id

    # Add and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(session_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream response with conversation history
    with st.chat_message("assistant"):
        history = st.session_state.messages[:-1]
        token_gen, sources = generate_response_with_sources(prompt, history)
        response = st.write_stream(token_gen)

    # Show sources in expander
    if sources:
        with st.expander("出典情報", expanded=False):
            for i, src in enumerate(sources, 1):
                source_name = src.metadata.get("source", "不明")
                section = src.metadata.get("section", "")
                label = f"**出典{i}**: {source_name}"
                if section:
                    label += f" - {section}"
                label += f" (類似度: {src.similarity:.2f})"
                st.markdown(label)
                st.caption(src.content[:200] + ("..." if len(src.content) > 200 else ""))

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(session_id, "assistant", response)

    # Auto-title on first exchange
    if len(st.session_state.messages) == 2:
        update_session_title(session_id, prompt[:30])
