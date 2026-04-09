"""LangGraph RAG pipeline with conversation history support."""

from typing import Annotated, Generator

from typing_extensions import TypedDict
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from lib.rag_chain import search_relevant_documents, RAG_SYSTEM_PROMPT


class RAGState(TypedDict):
    """State schema for the RAG LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    sources: list


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant documents for the latest user question."""
    question = state["messages"][-1].content
    result = search_relevant_documents(question)
    return {"context": result["context"], "sources": result["sources"]}


def generate(state: RAGState) -> dict:
    """Generate response using LLM with conversation history and RAG context."""
    system_msg = SystemMessage(
        content=RAG_SYSTEM_PROMPT.format(context=state["context"])
    )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=20,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    response = llm.invoke([system_msg] + trimmed)
    return {"messages": [response]}


def build_rag_graph():
    """Build and compile the RAG LangGraph."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def _build_messages(
    question: str, history: list[dict[str, str]]
) -> list[AnyMessage]:
    """Convert history dicts + question into LangChain messages."""
    messages: list[AnyMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def stream_response(
    question: str, history: list[dict[str, str]]
) -> Generator[str, None, None]:
    """Stream LLM tokens from the RAG graph for Streamlit consumption."""
    messages = _build_messages(question, history)

    graph = build_rag_graph()
    for chunk in graph.stream(
        {"messages": messages, "context": "", "sources": []},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            msg_chunk, metadata = chunk["data"]
            if msg_chunk.content and metadata.get("langgraph_node") == "generate":
                yield msg_chunk.content


def stream_response_with_sources(
    question: str, history: list[dict[str, str]]
) -> tuple[Generator[str, None, None], list]:
    """Stream LLM tokens and return sources from the RAG graph.

    Returns (token_generator, sources) tuple.
    Sources are fetched directly via search_relevant_documents,
    token streaming is delegated to stream_response.
    """
    result = search_relevant_documents(question)
    sources = result["sources"]
    return stream_response(question, history), sources
