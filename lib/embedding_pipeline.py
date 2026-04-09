"""Embedding pipeline: load documents, chunk, embed, store in Supabase."""

import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings

from lib.supabase_client import get_supabase_admin

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents(directory: str) -> list[Document]:
    """Load text/markdown/PDF files from a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: list[Document] = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "type": "pdf"},
                )
            )
        else:
            text = file_path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "type": file_path.suffix.lstrip("."),
                    },
                )
            )

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks.

    For Markdown (.md): 2-stage split using headers first, then size.
    For other formats: size-based split only.
    """
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
            # Stage 1: split by Markdown headers
            header_splits = md_header_splitter.split_text(doc.page_content)
            # Stage 2: split each section by size
            for section_doc in header_splits:
                section_name = (
                    section_doc.metadata.get("h2")
                    or section_doc.metadata.get("h1")
                    or ""
                )
                size_splits = size_splitter.split_text(section_doc.page_content)
                for i, text in enumerate(size_splits):
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": len(chunks),
                                "section": section_name,
                            },
                        )
                    )
        else:
            # Non-markdown: size-based split only
            splits = size_splitter.split_text(doc.page_content)
            for i, text in enumerate(splits):
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
    return chunks


def generate_embeddings(chunks: list[Document]) -> list[list[float]]:
    """Generate embeddings in batches of 100."""
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
    """Insert chunks and embeddings into Supabase documents table."""
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
    """CLI entry point."""
    from dotenv import load_dotenv

    load_dotenv(".env.local")

    dir_arg = "./docs"
    args = sys.argv[1:]
    if "--dir" in args:
        idx = args.index("--dir")
        dir_arg = args[idx + 1]

    print(f"Loading documents from: {dir_arg}")
    docs = load_documents(dir_arg)
    print(f"Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    print("Storing in Supabase...")
    store_in_supabase(chunks, embeddings)
    print("Done!")


if __name__ == "__main__":
    main()
