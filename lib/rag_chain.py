"""RAG検索ロジック: Embedding、Supabase検索、Reranking、プロンプト構築."""

from dataclasses import dataclass
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from lib.supabase_client import get_supabase_admin


RAG_SYSTEM_PROMPT = """あなたは社内ナレッジに基づいて質問に回答するアシスタントです。
以下のコンテキスト情報と、あなたの一般知識を組み合わせて、質問に丁寧に回答してください。

## 回答の優先順位:
1. コンテキストに直接的な回答がある場合 → コンテキストの情報を主な根拠として回答
2. コンテキストに部分的・関連する情報がある場合 → その範囲で回答し、不足部分は一般知識で補足
3. コンテキストに関連情報がない場合 → 一般知識で回答

## ルール:
- コンテキストから引用する場合は「」で囲んでください
- 質問の表記ゆれ（ひらがな・カタカナ・略語）は柔軟に解釈してください
- コンテキスト情報に基づく回答には【ナレッジベース】と明記してください
- 一般知識に基づく回答には【一般知識】と明記してください
- 両方を組み合わせる場合は、どの部分がどちらに基づくか区別してください

# コンテキスト:
{context}"""


@dataclass
class Source:
    content: str
    metadata: dict
    similarity: float


@lru_cache(maxsize=1)
def _get_embeddings():
    """キャッシュ済みの OpenAIEmbeddings インスタンスを返す."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """キャッシュ済みの Cross-Encoder モデルインスタンスを返す."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(
    question: str,
    sources: list[Source],
    top_k: int = 5,
) -> list[Source]:
    """Cross-Encoder で検索結果を再スコアリングし、上位 top_k 件を返す.

    Args:
        question: ユーザーのクエリ（リライト後）
        sources: 初回検索で取得した Source リスト
        top_k: Reranking 後に返す件数

    Returns:
        再スコアリング後の上位 top_k 件の Source リスト
    """
    if not sources or len(sources) <= top_k:
        return sources

    model = _get_cross_encoder()

    # Cross-Encoder はクエリとドキュメントのペアを入力として受け取る
    pairs = [[question, src.content] for src in sources]
    scores = model.predict(pairs)

    # スコアでソートし、上位 top_k 件を返す
    scored_sources = list(zip(scores, sources))
    scored_sources.sort(key=lambda x: x[0], reverse=True)

    return [
        Source(
            content=src.content,
            metadata=src.metadata,
            similarity=float(score),  # Reranking スコアで上書き
        )
        for score, src in scored_sources[:top_k]
    ]


def search_relevant_documents(
    question: str,
    match_threshold: float = 0.3,
    match_count: int = 10,
    use_hybrid: bool = True,
    metadata_filter: dict | None = None,
) -> dict:
    """ハイブリッド検索でドキュメントを取得する.

    Args:
        question: 検索クエリ
        match_threshold: 類似度の閾値
        match_count: 取得件数（Reranking 前の候補数）
        use_hybrid: ハイブリッド検索を使用するか
        metadata_filter: メタデータフィルタ条件（Phase 5 で使用）
    """
    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(question)

    supabase = get_supabase_admin()

    # メタデータフィルタ付きハイブリッド検索
    if use_hybrid and metadata_filter:
        result = (
            supabase.rpc(
                "match_documents_hybrid_filtered",
                {
                    "query_embedding": query_embedding,
                    "query_text": question,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                    "metadata_filter": metadata_filter,
                },
            ).execute()
        )
    elif use_hybrid:
        result = (
            supabase.rpc(
                "match_documents_hybrid",
                {
                    "query_embedding": query_embedding,
                    "query_text": question,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                },
            ).execute()
        )
    else:
        result = (
            supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                },
            ).execute()
        )

    docs = result.data or []

    sources = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        sources.append(
            Source(
                content=doc["content"],
                metadata=metadata,
                similarity=doc["similarity"],
            )
        )

    return {"sources": sources}


def format_sources_as_context(sources: list[Source]) -> str:
    """Source リストを出典ラベル付きコンテキスト文字列に整形する."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        source_name = src.metadata.get("source", "不明")
        section_name = src.metadata.get("section", "")
        label = f"[出典{i}: {source_name}"
        if section_name:
            label += f" - {section_name}"
        label += "]"
        context_parts.append(f"{label}\n{src.content}")
    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(
    question: str, context: str
) -> list[dict[str, str]]:
    """RAGコンテキスト付きのチャットメッセージを構築する."""
    system_message = RAG_SYSTEM_PROMPT.format(context=context)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
