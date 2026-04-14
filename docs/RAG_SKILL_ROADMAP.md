# RAG開発スキル レベル定義（Lv.1〜Lv.7）

## Context

現在のRagSmallTrainingプロジェクトは「中級（Lv.3〜4相当）」に位置する。ハイブリッド検索やクエリリライト等を実装済みだが、Reranking・Graph RAG・Agentic RAG・本格的な評価基盤など上級領域は未着手。転職・自社プロダクト・LLMプロダクト改善を見据え、TOEICのスコア帯のように明確なレベル定義を設け、現在地と次のマイルストーンを可視化する。

---

## レベル定義一覧

### Lv.1 — Naive RAG（入門）

**TOEICで例えると: 300〜400点 — 基本単語を知っている段階**

| 項目 | 内容 |
|------|------|
| **できること** | LangChain/LlamaIndexのチュートリアルを写経してRAGが動く |
| **検索** | 単純なベクトル検索（cosine similarity）のみ |
| **チャンキング** | 固定長分割（500文字ずつ等） |
| **Embedding** | OpenAI text-embedding-3-small をそのまま使用 |
| **LLM** | 1モデル固定（GPT-4o-mini等） |
| **評価** | 「動いたからOK」— 定量評価なし |
| **代表的な成果物** | PDFを読み込んで質問に答えるチャットボット |

---

### Lv.2 — Basic RAG（基礎）

**TOEICで例えると: 500〜600点 — 日常会話レベル**

| 項目　　　　　　　 | 内容　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------| --------------------------------------------------------------|
| **できること**　　 | チャンキング戦略を意識的に選択し、メタデータ付きで保存できる |
| **検索**　　　　　 | ベクトル検索 + 類似度閾値の調整　　　　　　　　　　　　　　　|
| **チャンキング**　 | Markdown見出し分割、オーバーラップ付き分割　　　　　　　　　 |
| **パイプライン**　 | 複数ノードのパイプライン（LangGraph等）　　　　　　　　　　　|
| **UI**　　　　　　 | ソース引用表示、類似度スコア表示　　　　　　　　　　　　　　 |
| **評価**　　　　　 | テストコードでユニットテスト（モック中心）　　　　　　　　　 |
| **代表的な成果物** | ソース付きチャットボット、セッション管理　　　　　　　　　　 |

---

### Lv.3 — Intermediate RAG（中級前半）⭐ 現在地

**TOEICで例えると: 650〜730点 — ビジネスで使える最低ライン**

| 項目　　　　　　　 | 内容　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------| ------------------------------------------------------------|
| **できること**　　 | ハイブリッド検索、クエリリライトなど検索品質の改善ができる |
| **検索**　　　　　 | ハイブリッド検索（ベクトル + BM25キーワード検索）　　　　　|
| **クエリ処理**　　 | クエリリライトノード（略語展開、タイポ修正）　　　　　　　 |
| **LLM**　　　　　　| 複数モデル切り替え対応（GPT-4o-mini / Gemini）　　　　　　 |
| **ストリーミング** | 単一グラフ実行でトークン + ソースを同時取得　　　　　　　　|
| **キャッシュ**　　 | LLM・グラフ・Embeddingの`lru_cache`　　　　　　　　　　　　|
| **DB**　　　　　　 | pgvector + GINインデックス、RLS　　　　　　　　　　　　　　|
| **テスト**　　　　 | TDD、カバレッジ90%+、モック活用　　　　　　　　　　　　　　|

**→ あなたのRagSmallTrainingはここ**

---

### Lv.4 — Intermediate RAG（中級後半）— 次の目標

**TOEICで例えると: 730〜800点 — 海外赴任の足切りライン**

| 項目　　　　　　　　　　　　　 | 習得すべき技術　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Reranking**　　　　　　　　　| Cross-Encoder（`cross-encoder/ms-marco-MiniLM-L-6-v2`）またはCohere Rerank APIで検索結果を再スコアリング。初回検索で20件取得→Rerankerで上位5件に絞る |
| **セマンティックチャンキング** | 固定長ではなく、文間の意味的類似度が閾値以下になった箇所で分割。内容の一貫性を保つ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **クエリ変換の高度化**　　　　 | HyDE（仮説的な回答を生成→その回答のEmbeddingで検索）、Multi-Query Expansion（3〜5の言い換えで並列検索→重複除去）　　　　　　　　　　　　　　　　　　 |
| **RAG評価フレームワーク**　　　| RAGAS導入。Context Recall / Context Precision / Faithfulness / Answer Relevancy の4指標を定量計測　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **メタデータフィルタリング**　 | 検索前にメタデータ（日付、カテゴリ、ソースタイプ）でプレフィルタ。不要なチャンクを検索対象から除外　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **実装目安**　　　　　　　　　 | 既存プロジェクトにReranking + RAGAS評価を追加する　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

### Lv.5 — Advanced RAG（上級前半）

**TOEICで例えると: 800〜860点 — 外資系で戦えるレベル**

| 項目 | 習得すべき技術 |
|------|------|
| **Graph RAG** | Neo4j等でエンティティ間の関係をグラフ化。マルチホップ推論（A→B→Cの連鎖的な質問）に対応。法務・医療・研究など関係性が重要なドメインで25%精度向上の報告あり |
| **Agentic RAG** | LLMが「何を検索するか」「検索結果が十分か」を自律判断。検索→評価→再検索のループを自動実行。LangGraphのconditional edgeで実装 |
| **マルチモーダルRAG** | テキスト + 画像 + 表 を統合検索。PDFの図表をVision LLMで解析してテキスト化→Embeddingに含める |
| **Fine-tuning** | Embedding モデルの fine-tuning（ドメイン特化のベクトル空間を構築）。sentence-transformersで自社データに最適化 |
| **本番監視** | LangSmith / LangFuse でリクエストごとのレイテンシ・トークン使用量・検索精度を可視化。A/Bテスト基盤 |
| **実装目安** | Agentic RAGで「検索→不足判定→追加検索」の自律ループを構築 |

---

### Lv.6 — Production RAG（上級後半）

**TOEICで例えると: 860〜940点 — ネイティブに近い運用力**

| 項目 | 習得すべき技術 |
|------|------|
| **大規模運用** | 50,000+チャンクの本番環境。検索レイテンシ400-700ms以内のSLA管理 |
| **コスト最適化** | LLMルーティング（簡単な質問→小さいモデル、複雑→大きいモデル）。キャッシュレイヤー（同一クエリの再利用） |
| **セキュリティ** | ユーザーごとのアクセス制御（RBACベースのRAG）。PII検出・マスキング |
| **CI/CDパイプライン** | RAG回帰テスト自動化。Embeddingモデル更新時の自動再インデックス |
| **マルチテナント** | 複数組織がそれぞれ独自のナレッジベースを持つSaaS構築 |
| **ガードレール** | 入力検証（プロンプトインジェクション防止）、出力検証（ハルシネーション検出・フィルタリング） |
| **実装目安** | マルチテナントRAG SaaSを構築し、実ユーザーに提供 |

---

### Lv.7 — Expert / Research（エキスパート）

**TOEICで例えると: 950〜990点 — 専門書を書けるレベル**

| 項目 | 内容 |
|------|------|
| **独自アーキテクチャ** | 既存フレームワークに頼らず、ドメイン最適なRAGアーキテクチャを設計 |
| **論文実装** | 最新のRAG論文（CRAG, Self-RAG, RAPTOR等）を読んで自プロダクトに適用 |
| **Embedding研究** | 独自Embeddingモデルの学習、Matryoshka Representation Learning等の最新手法 |
| **ベンチマーク構築** | ドメイン特化の評価データセットを自作し、定量比較 |
| **OSS貢献** | LangChain / LlamaIndex / RAGAS 等へのコントリビュート |
| **コミュニティ** | 技術ブログ・登壇・論文発表でRAGの知見を発信 |

---

## 現在地 → 次のステップ（Lv.3 → Lv.4 ロードマップ）

### 優先順位（効果/コスト比の高い順）

| 順番 | 施策 | 効果 | 難易度 | 推定期間 |
|------|------|------|--------|----------|
| 1 | **RAGAS評価基盤の導入** | 改善を定量計測できるようになる（全レベルアップの前提） | ★★☆ | 1-2週間 |
| 2 | **Cross-Encoder Reranking追加** | Precision大幅向上（最もコスパの良い精度改善） | ★★☆ | 1週間 |
| 3 | **セマンティックチャンキング** | チャンク品質向上→検索精度向上 | ★★★ | 1-2週間 |
| 4 | **HyDE / Multi-Query Expansion** | Recall改善（特に曖昧な質問への対応力） | ★★★ | 1-2週間 |
| 5 | **メタデータフィルタリング** | 検索の前段で不要チャンクを除外 | ★★☆ | 1週間 |

### Lv.4達成の目安

- 上記5つのうち3つ以上を実装
- RAGASで4指標を定量計測し、改善前後の比較データを出せる
- ポートフォリオとして「Naive RAG → Advanced RAGへの改善ジャーニー」を記事化できる

---

## 参考リソース

- [How to Become a RAG Specialist (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2024/12/rag-specialist/) — 16ステップのロードマップ
- [The 2025-26 RAG Project Blueprint (AI Fire)](https://www.aifire.co/p/the-2025-26-rag-project-blueprint-for-a-standout-ai-career) — ポートフォリオ向け10プロジェクト
- [RAG Zero to Hero Guide (GitHub)](https://github.com/KalyanKS-NLP/rag-zero-to-hero-guide/blob/main/RAG%20Basics/RAG_Roadmap.md) — 基礎→上級ロードマップ
- [Advanced RAG — Hybrid Search, Reranking & Knowledge Graphs](https://myengineeringpath.dev/genai-engineer/advanced-rag/) — 上級テクニック実装ガイド
- [RAG Is Not Dead: Advanced Patterns (DEV.to)](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo) — 2026年の最新パターン
- [【2025年完全版】RAGの教科書 (Zenn)](https://zenn.dev/microsoft/articles/rag_textbook) — 日本語の網羅的ガイド
- [RAG精度を引き上げる8つの鍵 (Arpable)](https://arpable.com/artificial-intelligence/rag/rag-performance-improvement-strategies/) — Graph RAG時代の実践ガイド
