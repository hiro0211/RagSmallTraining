"""LangGraph RAGパイプライン: HyDE + Multi-Query + Reranking + 会話履歴対応."""

from functools import lru_cache
from typing import Annotated, Generator

from typing_extensions import TypedDict
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from lib.llm import create_llm, DEFAULT_MODEL
from lib.rag_chain import (
    search_relevant_documents,
    rerank_documents,
    format_sources_as_context,
    Source,
    RAG_SYSTEM_PROMPT,
)


REWRITE_PROMPT = """あなたはユーザーの質問を検索用に書き換えるアシスタントです。
以下のルールで、元の質問の意図を保ったまま、検索に適した質問文に書き換えてください：

- 略語は正式名称に展開してください（例: ラグ → RAG (Retrieval-Augmented Generation)）
- ひらがなのタイポは正規化してください（例: べくとる → ベクトル）
- 曖昧・断片的な質問は完全な疑問文にしてください（例: ベクトルか → ベクトル検索とは何ですか）
- 元の質問が既に十分明確な場合は、そのまま返してください
- 出力は書き換え後の質問文のみ。説明や前置きは不要。

元の質問: {question}
"""

HYDE_PROMPT = """以下の質問に対して、想定される回答を日本語で200文字程度で生成してください。
正確でなくても構いません。検索用の仮説的な回答として使用します。
回答のみを出力してください。前置きや説明は不要です。

質問: {question}
"""

MULTI_QUERY_PROMPT = """以下の質問を、異なる表現で3つに言い換えてください。
元の質問の意図を保ちつつ、異なる語彙や視点で表現してください。
各言い換えを改行で区切って出力してください。番号や記号は不要です。

元の質問: {question}
"""


class RAGState(TypedDict):
    """RAG LangGraph のステートスキーマ."""

    messages: Annotated[list[AnyMessage], add_messages]
    rewritten_query: str
    hyde_context: str          # HyDE の仮説回答
    expanded_queries: list     # Multi-Query の言い換えリスト
    context: str
    sources: list              # 最終的なソースリスト
    hyde_sources: list         # HyDE 検索結果
    multi_query_sources: list  # Multi-Query 検索結果
    original_sources: list     # オリジナルクエリ検索結果
    model_id: str


# ---------------------------------------------------------------------------
# ノード関数
# ---------------------------------------------------------------------------

def rewrite_query(state: RAGState) -> dict:
    """ユーザークエリを検索用にリライトする."""
    question = state["messages"][-1].content
    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip() or question
    return {"rewritten_query": rewritten}


def hyde_query(state: RAGState) -> dict:
    """HyDE: 仮説的な回答を生成し、その回答のEmbeddingで検索する."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)

    # 仮説回答を生成
    prompt = HYDE_PROMPT.format(question=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    hyde_answer = response.content.strip()

    # 仮説回答で検索（仮説回答のEmbeddingが実際のドキュメントに近い）
    result = search_relevant_documents(hyde_answer, match_count=10)
    return {
        "hyde_context": hyde_answer,
        "hyde_sources": result["sources"],
    }


def multi_query_expand(state: RAGState) -> dict:
    """Multi-Query: 質問を3つに言い換えて並列検索する."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)

    # 言い換えクエリを生成
    prompt = MULTI_QUERY_PROMPT.format(question=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    expanded = [
        q.strip() for q in response.content.strip().split("\n") if q.strip()
    ][:3]  # 最大3つ

    # 各クエリで検索
    all_sources: list[Source] = []
    for eq in expanded:
        result = search_relevant_documents(eq, match_count=10)
        all_sources.extend(result["sources"])

    return {
        "expanded_queries": expanded,
        "multi_query_sources": all_sources,
    }


def retrieve(state: RAGState) -> dict:
    """オリジナルのリライト済みクエリでハイブリッド検索（10件取得）."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    result = search_relevant_documents(query, match_count=10)
    return {"original_sources": result["sources"]}


def merge_results(state: RAGState) -> dict:
    """HyDE + Multi-Query + オリジナルの検索結果を統合・重複除去する."""
    all_sources: list[Source] = []
    all_sources.extend(state.get("hyde_sources") or [])
    all_sources.extend(state.get("multi_query_sources") or [])
    all_sources.extend(state.get("original_sources") or [])

    # 重複除去（content ベース）
    seen_contents: set[str] = set()
    unique_sources: list[Source] = []
    for src in all_sources:
        # コンテンツの先頭200文字をキーとして重複判定
        content_key = src.content[:200]
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_sources.append(src)

    return {"sources": unique_sources}


def rerank(state: RAGState) -> dict:
    """Cross-Encoder で統合済み検索結果を再スコアリングし、上位5件に絞る."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    sources = state.get("sources", [])
    reranked = rerank_documents(query, sources, top_k=5)
    context = format_sources_as_context(reranked)
    return {"sources": reranked, "context": context}


def generate(state: RAGState) -> dict:
    """会話履歴とRAGコンテキストを使ってLLMで回答を生成."""
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

    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    response = llm.invoke([system_msg] + trimmed)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# グラフ構築
# ---------------------------------------------------------------------------

def build_rag_graph():
    """RAG LangGraph を構築・コンパイルする.

    パイプライン:
      START → rewrite_query
        → [hyde_query, multi_query_expand, retrieve] (並列)
        → merge_results → rerank(5件) → generate → END
    """
    graph = StateGraph(RAGState)

    # ノード登録
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hyde_query", hyde_query)
    graph.add_node("multi_query_expand", multi_query_expand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("merge_results", merge_results)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    # エッジ: rewrite_query → 3つの検索を並列実行
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "hyde_query")
    graph.add_edge("rewrite_query", "multi_query_expand")
    graph.add_edge("rewrite_query", "retrieve")

    # 3つの検索結果を merge_results に集約
    graph.add_edge("hyde_query", "merge_results")
    graph.add_edge("multi_query_expand", "merge_results")
    graph.add_edge("retrieve", "merge_results")

    # 統合 → Rerank → 生成
    graph.add_edge("merge_results", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    """キャッシュ済みのコンパイル済みRAGグラフを返す."""
    return build_rag_graph()


def _build_messages(
    question: str, history: list[dict[str, str]]
) -> list[AnyMessage]:
    """履歴辞書 + 質問を LangChain メッセージに変換."""
    messages: list[AnyMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages


# ---------------------------------------------------------------------------
# ストリーミング応答
# ---------------------------------------------------------------------------

def _initial_state(messages: list[AnyMessage], model_id: str) -> dict:
    """グラフの初期ステートを構築する."""
    return {
        "messages": messages,
        "rewritten_query": "",
        "hyde_context": "",
        "expanded_queries": [],
        "context": "",
        "sources": [],
        "hyde_sources": [],
        "multi_query_sources": [],
        "original_sources": [],
        "model_id": model_id,
    }


def stream_response(
    question: str, history: list[dict[str, str]], model_id: str = ""
) -> Generator[str, None, None]:
    """RAGグラフからLLMトークンをストリーミングする."""
    messages = _build_messages(question, history)
    graph = get_compiled_graph()

    for chunk in graph.stream(
        _initial_state(messages, model_id),
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            msg_chunk, metadata = chunk["data"]
            if msg_chunk.content and metadata.get("langgraph_node") == "generate":
                yield msg_chunk.content


def stream_response_with_sources(
    question: str, history: list[dict[str, str]], model_id: str = ""
) -> tuple[Generator[str, None, None], list]:
    """LLMトークンをストリーミングし、ソース情報も返す.

    マルチモードストリーミング ["updates", "messages"] を使用して、
    rerank ノードからソースを取得し、generate ノードからトークンを
    ストリーミングする。1回のグラフ実行で完結。

    Returns:
        (token_generator, sources) タプル。
        sources はジェネレータの消費に伴って追加される。
    """
    messages = _build_messages(question, history)
    graph = get_compiled_graph()
    sources: list = []

    def _generator():
        for chunk in graph.stream(
            _initial_state(messages, model_id),
            stream_mode=["updates", "messages"],
            version="v2",
        ):
            if chunk["type"] == "updates":
                payload = chunk["data"]
                # rerank ノードから Reranking 後のソースを取得
                if "rerank" in payload and payload["rerank"].get("sources"):
                    sources.extend(payload["rerank"]["sources"])
            elif chunk["type"] == "messages":
                msg_chunk, metadata = chunk["data"]
                if msg_chunk.content and metadata.get("langgraph_node") == "generate":
                    yield msg_chunk.content

    return _generator(), sources
