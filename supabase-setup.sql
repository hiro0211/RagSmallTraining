-- =============================================
-- RAG Small: Supabase pgvector セットアップ
-- Supabase SQL Editor で実行してください
-- =============================================

-- 1. pgvector 拡張を有効化
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. documents テーブル作成
CREATE TABLE IF NOT EXISTS documents (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  embedding vector(1536),
  created_at timestamptz DEFAULT now()
);

-- 3. ベクトル類似検索用 RPC 関数
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 5
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
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 4. HNSW インデックス作成（高速ベクトル検索用）
CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 5. RLS (Row Level Security) ポリシー
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- サービスロールキーからのフルアクセスを許可
CREATE POLICY "Service role full access"
  ON documents
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- anon キーからの読み取りのみ許可
CREATE POLICY "Anon read access"
  ON documents
  FOR SELECT
  TO anon
  USING (true);

-- =============================================
-- チャットセッション管理テーブル
-- =============================================

-- 6. chat_sessions テーブル
CREATE TABLE IF NOT EXISTS chat_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL DEFAULT '新しい会話',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- 7. chat_messages テーブル
CREATE TABLE IF NOT EXISTS chat_messages (
  id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('user', 'assistant')),
  content text NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- 8. メッセージ検索用インデックス
CREATE INDEX IF NOT EXISTS chat_messages_session_idx
  ON chat_messages(session_id, created_at);

-- 9. RLS ポリシー（チャットテーブル）
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access"
  ON chat_sessions
  FOR ALL
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access"
  ON chat_messages
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- =============================================
-- ハイブリッド検索（ベクトル + キーワード全文検索）
-- =============================================

-- 10. GIN インデックス（全文検索用）
CREATE INDEX IF NOT EXISTS documents_content_tsvector_idx
  ON documents
  USING gin (to_tsvector('simple', content));

-- 11. ハイブリッド検索 RPC 関数
CREATE OR REPLACE FUNCTION match_documents_hybrid(
  query_embedding vector(1536),
  query_text text,
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 5,
  vector_weight float DEFAULT 0.7,
  keyword_weight float DEFAULT 0.3
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
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
