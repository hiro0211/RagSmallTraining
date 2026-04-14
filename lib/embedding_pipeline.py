"""Embeddingパイプライン: ドキュメント読み込み、チャンク分割、Embedding生成、Supabase格納."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings

from lib.supabase_client import get_supabase_admin

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# メタデータのデフォルト値
DEFAULT_SOURCE_TYPE = "manual"
DEFAULT_CATEGORY = "general"
DEFAULT_IMPORTANCE = "medium"
DEFAULT_LANGUAGE = "ja"


def _build_metadata(file_path: Path, extra: dict | None = None) -> dict:
    """ファイルパスから拡張メタデータを構築する.

    メタデータ項目:
      - source: ファイル名
      - type: 拡張子
      - source_type: ドキュメント種別（デフォルト: manual）
      - created_at: ファイルの更新日時（ISO 8601）
      - category: ナレッジカテゴリ（デフォルト: general）
      - importance: 重要度（デフォルト: medium）
      - language: 言語（デフォルト: ja）
    """
    stat = file_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    metadata = {
        "source": file_path.name,
        "type": file_path.suffix.lstrip("."),
        "source_type": DEFAULT_SOURCE_TYPE,
        "created_at": mtime,
        "category": DEFAULT_CATEGORY,
        "importance": DEFAULT_IMPORTANCE,
        "language": DEFAULT_LANGUAGE,
    }
    if extra:
        metadata.update(extra)
    return metadata


def load_documents(
    directory: str, metadata_overrides: dict | None = None
) -> list[Document]:
    """ディレクトリからテキスト/Markdown/PDFファイルを読み込む."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")

    docs: list[Document] = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        metadata = _build_metadata(file_path, metadata_overrides)

        if file_path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append(
                Document(page_content=text, metadata=metadata)
            )
        else:
            text = file_path.read_text(encoding="utf-8")
            docs.append(
                Document(page_content=text, metadata=metadata)
            )

    return docs


def chunk_documents(
    docs: list[Document], use_semantic: bool = False
) -> list[Document]:
    """ドキュメントをチャンクに分割する.

    Args:
        docs: 分割対象のドキュメントリスト
        use_semantic: True の場合セマンティックチャンキングを使用

    Markdown (.md) の場合:
      1. 見出し（h1, h2）で分割
      2. 各セクションをさらにサイズベースまたはセマンティックで分割

    その他の場合:
      サイズベースまたはセマンティックで分割
    """
    if use_semantic:
        return _chunk_documents_semantic(docs)
    return _chunk_documents_fixed(docs)


def _chunk_documents_fixed(docs: list[Document]) -> list[Document]:
    """固定長チャンキング（従来方式）."""
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )

    chunks: list[Document] = []
    for doc in docs:
        is_markdown = doc.metadata.get("type") == "md"

        if is_markdown:
            header_splits = md_header_splitter.split_text(doc.page_content)
            for section_doc in header_splits:
                section_name = (
                    section_doc.metadata.get("h2")
                    or section_doc.metadata.get("h1")
                    or ""
                )
                size_splits = size_splitter.split_text(section_doc.page_content)
                for text in size_splits:
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(chunks),
                                "section": section_name,
                                "chunking_method": "fixed",
                            },
                        )
                    )
        else:
            splits = size_splitter.split_text(doc.page_content)
            for i, text in enumerate(splits):
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "chunking_method": "fixed",
                        },
                    )
                )
    return chunks


def _chunk_documents_semantic(docs: list[Document]) -> list[Document]:
    """セマンティックチャンキング（SemanticChunker使用）."""
    from langchain_experimental.text_splitters import SemanticChunker

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # 上位5%の類似度低下箇所で分割
    )
    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )

    chunks: list[Document] = []
    for doc in docs:
        is_markdown = doc.metadata.get("type") == "md"

        if is_markdown:
            # Markdown: 見出し分割後にセマンティックチャンキング
            header_splits = md_header_splitter.split_text(doc.page_content)
            for section_doc in header_splits:
                section_name = (
                    section_doc.metadata.get("h2")
                    or section_doc.metadata.get("h1")
                    or ""
                )
                # テキストが短すぎる場合はそのまま1チャンクにする
                if len(section_doc.page_content) < 100:
                    chunks.append(
                        Document(
                            page_content=section_doc.page_content,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(chunks),
                                "section": section_name,
                                "chunking_method": "semantic",
                            },
                        )
                    )
                    continue

                semantic_splits = semantic_chunker.split_text(
                    section_doc.page_content
                )
                for text in semantic_splits:
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(chunks),
                                "section": section_name,
                                "chunking_method": "semantic",
                            },
                        )
                    )
        else:
            # 非Markdown: 直接セマンティックチャンキング
            if len(doc.page_content) < 100:
                chunks.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_index": 0,
                            "chunking_method": "semantic",
                        },
                    )
                )
                continue

            semantic_splits = semantic_chunker.split_text(doc.page_content)
            for i, text in enumerate(semantic_splits):
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "chunking_method": "semantic",
                        },
                    )
                )
    return chunks


def generate_embeddings(chunks: list[Document]) -> list[list[float]]:
    """100件ずつバッチでEmbeddingを生成する."""
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        batch_embeddings = embeddings_model.embed_documents(texts)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def store_in_supabase(
    chunks: list[Document], embeddings: list[list[float]]
) -> None:
    """チャンクとEmbeddingをSupabaseのdocumentsテーブルに格納する."""
    supabase = get_supabase_admin()
    rows = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embeddings[i],
        }
        for i, chunk in enumerate(chunks)
    ]

    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        supabase.table("documents").insert(batch).execute()


def main():
    """CLIエントリーポイント."""
    from dotenv import load_dotenv

    load_dotenv(".env.local")

    args = sys.argv[1:]

    # ディレクトリ指定
    dir_arg = "./docs"
    if "--dir" in args:
        idx = args.index("--dir")
        dir_arg = args[idx + 1]

    # セマンティックチャンキング切り替え
    use_semantic = "--semantic" in args

    # メタデータ上書き
    metadata_overrides = {}
    if "--source-type" in args:
        idx = args.index("--source-type")
        metadata_overrides["source_type"] = args[idx + 1]
    if "--category" in args:
        idx = args.index("--category")
        metadata_overrides["category"] = args[idx + 1]

    chunking_label = "セマンティック" if use_semantic else "固定長"
    print(f"ドキュメント読み込み: {dir_arg}")
    docs = load_documents(dir_arg, metadata_overrides or None)
    print(f"読み込み完了: {len(docs)} ファイル")

    print(f"チャンク分割中（{chunking_label}）...")
    chunks = chunk_documents(docs, use_semantic=use_semantic)
    print(f"チャンク作成完了: {len(chunks)} チャンク")

    print("Embedding生成中...")
    embeddings = generate_embeddings(chunks)
    print(f"Embedding生成完了: {len(embeddings)} 件")

    print("Supabaseに格納中...")
    store_in_supabase(chunks, embeddings)
    print("完了!")


if __name__ == "__main__":
    main()
