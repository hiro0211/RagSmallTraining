# 変更影響ログ

## CHG-001: RAG Lv.4 刷新計画策定

- 変更内容: RAGパイプラインの5施策（RAGAS評価 / Reranking / セマンティックチャンキング / HyDE+Multi-Query / メタデータフィルタリング）の実装計画を策定
- 影響を受ける文書:
  - `ARCHITECTURE.md` — パイプラインフロー図、ノード構成の更新が必要
  - `docs/RAG_SKILL_ROADMAP.md` — Lv.4 達成状況の更新
  - `aidlc-docs/decision_log.md` — 各Phase完了時に決定事項を追記
- 影響を受けるコード:
  - `lib/evaluator.py` — RAGAS 4指標の計測ロジック追加
  - `lib/rag_chain.py` — `rerank_documents()`, メタデータフィルタ引数追加
  - `lib/embedding_pipeline.py` — `semantic_chunk_documents()`, メタデータ拡張
  - `lib/graph.py` — `hyde_query()`, `multi_query_expand()`, `rerank` ノード追加
  - `supabase-setup.sql` — `match_documents_hybrid` のメタデータフィルタ対応
  - `requirements.txt` / `pyproject.toml` — `sentence-transformers`, `langchain-experimental` 追加
  - `tests/` — 各新機能のユニットテスト
- 再レビューが必要な箇所:
  - `lib/graph.py` の LangGraph パイプライン構成（ノード追加による影響）
  - `supabase-setup.sql` のRPC関数（既存の `match_documents_hybrid` との互換性）
  - `app.py` のUI（メタデータフィルタのUI追加が必要になる可能性）

## CHG-002: Supabase スキーマ変更（Phase 5 用）

- 変更内容:
  - `documents_metadata_idx` GINインデックス追加（`metadata jsonb_path_ops`）
  - `match_documents_hybrid_filtered()` RPC関数追加（メタデータフィルタ付きハイブリッド検索）
- 影響を受ける文書:
  - `supabase-setup.sql` — 追記済み
  - `aidlc-docs/migration_lv4.sql` — マイグレーション用SQL作成済み
- 影響を受けるコード:
  - `lib/rag_chain.py` — Phase 5 で `match_documents_hybrid_filtered` を呼び出すよう変更予定
- 再レビューが必要な箇所:
  - 既存の `match_documents_hybrid` との互換性（新関数は `metadata_filter=NULL` で同等動作）
  - GINインデックスの `jsonb_path_ops` が `@>` 演算子のみ対応（`?`, `?|`, `?&` は非対応）

## CHG-003: テストデータ拡充

- 変更内容: `docs/test.md` にRAG技術の詳細説明を追加（ハイブリッド検索、チャンキング戦略、Reranking、クエリ変換、RAGAS、メタデータフィルタリング）
- 影響を受ける文書: `docs/test.md`
- 影響を受けるコード: なし（ナレッジベースの拡充のみ）
- 再レビューが必要な箇所: Embedding パイプラインの再実行が必要（新しいコンテンツをベクトル化してSupabaseに格納）
