import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest


class TestLoadDocuments:
    def test_loads_text_files(self, tmp_path):
        (tmp_path / "test.txt").write_text("Hello world", encoding="utf-8")
        from lib.embedding_pipeline import load_documents

        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"
        assert docs[0].metadata["source"] == "test.txt"

    def test_loads_markdown_files(self, tmp_path):
        (tmp_path / "test.md").write_text("# Title\nContent", encoding="utf-8")
        from lib.embedding_pipeline import load_documents

        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert "Title" in docs[0].page_content

    def test_ignores_unsupported_files(self, tmp_path):
        (tmp_path / "test.txt").write_text("valid", encoding="utf-8")
        (tmp_path / "test.jpg").write_bytes(b"invalid")
        from lib.embedding_pipeline import load_documents

        docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_loads_pdf_files(self, tmp_path):
        from lib.embedding_pipeline import load_documents

        with patch("pypdf.PdfReader") as MockReader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF content here"
            MockReader.return_value.pages = [mock_page]
            (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 fake")
            docs = load_documents(str(tmp_path))
            assert len(docs) == 1
            assert docs[0].page_content == "PDF content here"
            assert docs[0].metadata["type"] == "pdf"

    def test_raises_on_missing_directory(self):
        from lib.embedding_pipeline import load_documents

        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path")


class TestChunkDocuments:
    def test_splits_long_document(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        doc = Document(page_content="あ" * 2000, metadata={"source": "test.md"})
        chunks = chunk_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 500

    def test_preserves_metadata(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        doc = Document(
            page_content="あ" * 2000, metadata={"source": "test.md", "type": "md"}
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"

    def test_uses_japanese_separators(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        text = "最初の段落です。" + "あ" * 900 + "。次の段落が始まります。" + "い" * 900
        doc = Document(page_content=text, metadata={"source": "test.md"})
        chunks = chunk_documents([doc])
        assert len(chunks) >= 2

    def test_markdown_splits_by_headers(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        md = "# セクション1\nセクション1の内容です。\n\n## セクション2\nセクション2の内容です。\n\n## セクション3\nセクション3の内容です。"
        doc = Document(page_content=md, metadata={"source": "test.md", "type": "md"})
        chunks = chunk_documents([doc])
        assert len(chunks) == 3

    def test_markdown_chunks_have_section_metadata(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        md = "# タイトル\n本文の内容\n\n## サブセクション\nサブの内容"
        doc = Document(page_content=md, metadata={"source": "test.md", "type": "md"})
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "section" in chunk.metadata

    def test_non_markdown_uses_size_splitter_only(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        doc = Document(
            page_content="あ" * 1000,
            metadata={"source": "test.txt", "type": "txt"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) >= 2
        # txt should NOT have section metadata
        for chunk in chunks:
            assert "section" not in chunk.metadata

    def test_chunk_size_is_500(self):
        from lib.embedding_pipeline import chunk_documents
        from langchain_core.documents import Document

        doc = Document(
            page_content="あ" * 2000,
            metadata={"source": "test.txt", "type": "txt"},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= 500


class TestGenerateEmbeddings:
    def test_generates_embeddings(self):
        from lib.embedding_pipeline import generate_embeddings
        from langchain_core.documents import Document

        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]
        chunks = [
            Document(page_content="テスト1", metadata={}),
            Document(page_content="テスト2", metadata={}),
        ]

        with patch("lib.embedding_pipeline.OpenAIEmbeddings") as MockEmbed:
            instance = MockEmbed.return_value
            instance.embed_documents.return_value = mock_embeddings
            result = generate_embeddings(chunks)

        assert len(result) == 2
        assert len(result[0]) == 1536

    def test_batches_large_input(self):
        from lib.embedding_pipeline import generate_embeddings
        from langchain_core.documents import Document

        chunks = [Document(page_content=f"doc{i}", metadata={}) for i in range(150)]

        with patch("lib.embedding_pipeline.OpenAIEmbeddings") as MockEmbed:
            instance = MockEmbed.return_value
            instance.embed_documents.return_value = [[0.1] * 1536] * 100
            # Second call returns 50
            instance.embed_documents.side_effect = [
                [[0.1] * 1536] * 100,
                [[0.2] * 1536] * 50,
            ]
            result = generate_embeddings(chunks)

        assert len(result) == 150
        assert instance.embed_documents.call_count == 2


class TestStoreInSupabase:
    def test_inserts_into_supabase(self):
        from lib.embedding_pipeline import store_in_supabase
        from langchain_core.documents import Document

        chunks = [
            Document(page_content="テスト", metadata={"source": "test.md"}),
        ]
        embeddings = [[0.1] * 1536]

        with patch("lib.embedding_pipeline.get_supabase_admin") as mock_admin:
            mock_table = MagicMock()
            mock_table.insert.return_value.execute.return_value = MagicMock()
            mock_admin.return_value.table.return_value = mock_table

            store_in_supabase(chunks, embeddings)

            mock_admin.return_value.table.assert_called_with("documents")
            mock_table.insert.assert_called_once()
