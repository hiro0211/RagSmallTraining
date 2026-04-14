# 決定ログ

## DEC-001: RAG Lv.4 刷新 — 技術選定

- 日付: 2026-04-14
- 決定事項:
  1. Reranking: ローカル Cross-Encoder（`cross-encoder/ms-marco-MiniLM-L-6-v2`）を採用
  2. セマンティックチャンキング: `langchain_experimental.text_splitters.SemanticChunker` を使用
  3. クエリ変換: HyDE + Multi-Query を並列実行し、結果を統合後 Rerank で絞る設計
  4. メタデータフィルタリング: `source_type`, `created_at`, `category`, `importance`, `language` を追加
  5. 評価基盤: RAGAS 4指標（Context Recall / Context Precision / Faithfulness / Answer Relevancy）
- 根拠:
  1. ローカル実行はコスト不要、外部API依存なし。ユーザー要望
  2. 学習コスト低、LangChain エコシステムとの親和性。ユーザー要望
  3. HyDE（仮説回答のEmbeddingで検索）+ Multi-Query（言い換え並列検索）を同時実行し、Recall を最大化
  4. RAGスペシャリスト観点での推奨。Sales Metrics Engine の将来要件（テナント分離、CRMデータ等）を見据えた設計
  5. pyproject.toml に `ragas>=0.4.3` 既存。全改善の定量計測基盤として最優先
- 承認者: （レビュー待ち）
- 影響範囲:
  - `lib/evaluator.py` — RAGAS 4指標の計測ロジック追加
  - `lib/rag_chain.py` — Reranking関数、メタデータフィルタ追加
  - `lib/embedding_pipeline.py` — セマンティックチャンキング追加、メタデータ拡張
  - `lib/graph.py` — HyDE / Multi-Query / Rerank ノード追加、パイプライン再構成
  - `supabase-setup.sql` — メタデータフィルタ対応RPC関数拡張
  - `requirements.txt` / `pyproject.toml` — 依存パッケージ追加
  - `ARCHITECTURE.md` — アーキテクチャ文書更新
  - `tests/` — 各機能のテスト追加

---

## DEC-002: Reranking 取得件数

- 日付: 2026-04-14
- 決定事項: 初回検索の取得件数を10件とし、Cross-Encoder Rerank 後に上位5件に絞る
- 根拠:
  - 20件だとCPU環境で1〜3秒のレイテンシが発生する
  - 10件なら約0.5〜1.5秒に短縮でき、ユーザー体感への影響を抑えられる
  - 現在の5件から2倍の候補を確保するため、十分な精度改善が見込める
  - RAGAS で効果計測後、精度が不足すれば20件に増やす余地を残す
- 承認者: ユーザー承認済み
- 影響範囲:
  - `lib/rag_chain.py` — `match_count` デフォルト値を 5 → 10 に変更
  - `lib/graph.py` — Rerank ノードの入力件数
  - `aidlc-docs/rag_lv4_implementation_plan.md` — Phase 2 の記述更新

---

## DEC-003: セマンティックチャンキング閾値方式

- 日付: 2026-04-14
- 決定事項: SemanticChunker の `breakpoint_threshold_type` を `percentile` 方式（上位5%で分割）に設定
- 根拠:
  - ドキュメントの内容に応じて自動調整されるため、汎用性が高い
  - 固定値方式はドキュメント種類ごとに最適値が異なり、手動調整が必要
  - RAGAS 評価で効果計測後、必要に応じてパーセンテージや方式を調整可能
- 承認者: ユーザー承認済み
- 影響範囲:
  - `lib/embedding_pipeline.py` — SemanticChunker の設定値

---

## DEC-004: RAGAS テストデータセットの構築方針

- 日付: 2026-04-14
- 決定事項: `docs/test.md` ベースの自動生成。RAGの詳細な技術説明を `test.md` に追加
- 根拠:
  - 既存の `test.md` をナレッジベースとして活用し、そこからQ&Aを自動生成
  - RAG技術の詳細説明（ハイブリッド検索、チャンキング戦略、Reranking、クエリ変換、RAGAS、メタデータフィルタリング）を追加し、テストデータの多様性を確保
- 承認者: ユーザー承認済み
- 影響範囲:
  - `docs/test.md` — RAG技術の詳細説明を追加済み
  - `lib/evaluator.py` — テストデータセット自動生成ロジック
