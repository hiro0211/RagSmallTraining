"""RAG search logic: embed query, search Supabase, build prompt."""

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
    """Return a cached OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def search_relevant_documents(
    question: str,
    match_threshold: float = 0.3,
    match_count: int = 5,
    use_hybrid: bool = True,
) -> dict:
    """Search for relevant documents using vector similarity or hybrid search."""
    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(question)

    supabase = get_supabase_admin()

    if use_hybrid:
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

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.get("metadata", {})
        source_name = metadata.get("source", "不明")
        section_name = metadata.get("section", "")
        label = f"[出典{i}: {source_name}"
        if section_name:
            label += f" - {section_name}"
        label += "]"
        context_parts.append(f"{label}\n{doc['content']}")
        sources.append(
            Source(
                content=doc["content"],
                metadata=metadata,
                similarity=doc["similarity"],
            )
        )

    context = "\n\n---\n\n".join(context_parts)
    return {"context": context, "sources": sources}


def build_rag_prompt(
    question: str, context: str
) -> list[dict[str, str]]:
    """Build chat messages with RAG context."""
    system_message = RAG_SYSTEM_PROMPT.format(context=context)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
