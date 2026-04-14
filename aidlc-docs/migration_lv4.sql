-- =============================================
-- RAG Lv.4 刷新: マイグレーションSQL
-- 既存の Supabase 環境に対して実行してください
-- 既存テーブル・関数はそのまま維持されます
-- =============================================

-- =============================================
-- 1. メタデータ用 GIN インデックス追加
-- =============================================
CREATE INDEX IF NOT EXISTS documents_metadata_idx
  ON documents
  USING gin (metadata jsonb_path_ops);

-- =============================================
-- 2. メタデータフィルタ付きハイブリッド検索 RPC 関数
-- =============================================
CREATE OR REPLACE FUNCTION match_documents_hybrid_filtered(
  query_embedding vector(1536),
  query_text text,
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 10,
  vector_weight float DEFAULT 0.7,
  keyword_weight float DEFAULT 0.3,
  metadata_filter jsonb DEFAULT NULL
)
RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    (
      vector_weight * (1 - (d.embedding <=> query_embedding))
      + keyword_weight * ts_rank(
          to_tsvector('simple', d.content),
          plainto_tsquery('simple', query_text)
        )
    )::float AS similarity
  FROM documents d
  WHERE (1 - (d.embedding <=> query_embedding)) > match_threshold
    AND (
      metadata_filter IS NULL
      OR d.metadata @> metadata_filter
    )
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
