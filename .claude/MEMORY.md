## 2026-04-09

### 作業内容
- Python + Streamlit + LangChain でRAGスモール版を構築
- 当初 Next.js で開始したが、ユーザーの要望で Streamlit に変更
- TDD で全フェーズ���装完了（23テスト、カバレッジ100%）

### 完了フェーズ
- Phase 0: Python プロジェクト初期化（uv, pyproject.toml, pytest）
- Phase 1: Supabase pgvector セットアップ SQL 生成（MCP未接続のためSQL手動実行が必要）
- Phase 2: Embedding パイプライン（load → chunk → embed → store）
- Phase 3: RAG 検索ロジック（ベクトル���索 + プロンプト構築）
- Phase 4: Streamlit チャット UI + ストリーミング応答
- Phase 5: テスト用ドキュメント作成、テスト全通過

### ユーザーが対応すべき残タスク
1. OpenAI API キー取得 → .env.local に記入
2. Supabase プロジェクト作成
3. supabase-setup.sql を Supabase SQL Editor で実行
4. .env.local に Supabase URL/キーを記入
5. `uv run python lib/embedding_pipeline.py --dir ./docs` で Embedding 実行
6. `uv run streamlit run app.py` で動作確認

### 変更ファ���ル一覧
- `pyproject.toml`, `.gitignore`, `.env.local`, `.env.example`
- `lib/supabase_client.py`, `lib/embedding_pipeline.py`, `lib/rag_chain.py`, `lib/chat.py`
- `app.py` (Streamlit UI)
- `tests/test_supabase_client.py`, `tests/test_embedding_pipeline.py`, `tests/test_rag_chain.py`, `tests/test_app.py`
- `supabase-setup.sql`, `docs/test.md`, `USER_TASKS.md`

## 2026-04-10 モデル切り替え機能（GPT-4o-mini ⇔ Gemini 2.5 Flash）

### 作業内容
- LLMファクトリ `lib/llm.py` を新規追加（`create_llm`, `get_available_models`, `DEFAULT_MODEL`）
- `ChatOpenAI` と `ChatGoogleGenerativeAI` を `model_id` で切り替える薄い抽象
- `lib/graph.py` の `RAGState` に `model_id` フィールド追加、`generate` ノードで `create_llm` に委譲
- `stream_response` / `stream_response_with_sources` に `model_id` 引数追加
- `lib/chat.py` の `generate_response` / `generate_response_with_sources` に `model_id` パラメータ追加
- `app.py` にモデルセレクター（`st.selectbox`, `label_visibility="collapsed"`）をタイトル直下に配置
- 依存関係: `pyproject.toml` / `requirements.txt` に `langchain-google-genai>=2.0.0` 追加
- Embeddings は OpenAI (`text-embedding-3-small`) のままで再インデックス不要
- TDD 厳守（Red → Green → Refactor）、全77テストパス、カバレッジ 90.26%

### ユーザーが対応すべき残タスク
1. `.env.local` に `GOOGLE_API_KEY=your-key` を追加
2. 必要なら `uv pip install langchain-google-genai` を実行（既にインストール済みの可能性あり）
3. `streamlit run app.py` でモデル切り替え動作を確認

### 変更ファイル一覧
- `lib/llm.py` （新規）
- `tests/test_llm.py` （新規）
- `lib/graph.py`, `lib/chat.py`, `app.py`
- `tests/test_graph.py`, `tests/test_app.py`
- `pyproject.toml`, `requirements.txt`

## 2026-04-10 RAG柔軟性改善（閾値緩和 + プロンプト緩和 + クエリ前処理）

### 作業内容
短い・曖昧・タイポ質問（「ベクトルか」「べくとる」「ラグって何ですか」）が常に「ナレッジベースに含まれていません」と拒否される問題に対処：

1. **閾値緩和**: `search_relevant_documents` の `match_threshold` default を 0.5 → 0.3
   - `supabase-setup.sql` の `match_documents` 関数 default も同様に更新
2. **プロンプト緩和**: `RAG_SYSTEM_PROMPT` を書き換え
   - 「のみを根拠に」→「を主な根拠に」
   - 「推測や補完は禁止」を削除
   - 「表記ゆれ（ひらがな・カタカナ・略語）は柔軟に解釈」を追加
   - 「部分的な情報がある場合」の段階的回答を追加
3. **`rewrite_query` ノード追加** (LangGraph):
   - 新パイプライン: `START → rewrite_query → retrieve → generate → END`
   - LLMで質問を正規化・展開（略語展開・ひらがなタイポ修正・曖昧クエリの具体化）
   - 空文字を返した場合は元の質問にフォールバック
   - `RAGState` に `rewritten_query: str` フィールド追加
   - `retrieve` は `state.get("rewritten_query") or state["messages"][-1].content` で fallback
4. **`stream_response_with_sources`**: ソース即時表示のためリライト後クエリで検索（LLM呼び出しが2回/ターンに増加）

TDD 厳守（Red → Green）、全85テストパス、カバレッジ 90.75%。

### 影響
- LLM 呼び出しが 2回/ターン（rewrite + generate）に増加 → `gpt-4o-mini` で追加コスト ~$0.0001/ターン
- 閾値 0.3 により無関係ドキュメントが混ざる可能性 → プロンプトの「主な根拠」で LLM が取捨選択

### ユーザーが対応すべき残タスク
- Supabase SQL Editor で `supabase-setup.sql` の `match_documents` 関数を再実行（任意、lib 側から明示的に渡しているのでDBの既存 default 0.5 のままでも動作する）
- Streamlit 手動検証:
  1. 「ベクトルか」→ ベクトル検索の説明
  2. 「べくとる」→ ベクトル検索の説明
  3. 「ラグって何ですか」→ RAG の技術概要
  4. 「ベクトル検索とは？」→ 既存動作を維持

### 変更ファイル一覧
- `lib/rag_chain.py` - 閾値 0.3 + プロンプト緩和
- `lib/graph.py` - `REWRITE_PROMPT`, `rewrite_query` ノード, `RAGState.rewritten_query`, `retrieve` の fallback, `build_rag_graph` のエッジ更新, `stream_response_with_sources` の LLM リライト
- `supabase-setup.sql` - `match_documents` default 0.3
- `tests/test_rag_chain.py` - `test_default_threshold_is_0_3`, `test_prompt_allows_flexible_interpretation`
- `tests/test_graph.py` - `TestRewriteQueryNode`, `TestRetrieveUsesRewrittenQuery`, `TestGraphTopology` 追加、`TestStreamResponse` の sources テストを `create_llm` モック対応

## 2026-04-11 応答速度最適化（二重実行解消・グラフキャッシュ・rewriteスキップ）

### 作業内容
TDD（Red→Green）で3つの最適化を実施：

1. **二重実行の解消（最重要）**: `stream_response_with_sources` を `graph.stream` 1回実行に変更
   - 旧: LLM rewrite + DB検索（1回目）→ `stream_response` 内でグラフ全実行（2回目）
   - 新: `stream_mode=["messages", "updates"]` で1回のgraph.streamから sources（retrieveノードのupdates）とトークン（generateノードのmessages）を同時取得
   - `sources` リストはクロージャ経由でmutableに共有、generator消費後に呼び元で参照可能

2. **グラフのモジュールレベルキャッシュ**: `get_compiled_graph()` 追加
   - `_compiled_graph = None` モジュールレベル変数でキャッシュ
   - `stream_response` / `stream_response_with_sources` 両方で使用
   - `build_rag_graph()` はプロセス起動時に1回だけ呼ばれる

3. **`rewrite_query` の条件付きスキップ**: 明確な質問ではLLM呼び出しをスキップ
   - 条件: `len(question) >= 10` かつ `？/? /ですか/でしょうか` のいずれかを含む
   - 短い・曖昧・略語質問（べくとる、ラグって何ですか等）は引き続きLLMでリライト

### テスト
- 新規12テスト追加（`TestGetCompiledGraph`, `TestRewriteQuerySkip`, `TestStreamResponseWithSourcesSingleGraph`）
- 既存 `TestStreamResponse` の `@patch("lib.graph.build_rag_graph")` を `get_compiled_graph` に更新
- 旧 `stream_response_with_sources` テスト2本を削除（実装変更に合わせて新しいクラスに置換）
- 全97テストパス、カバレッジ 91.19%（lib/graph.py 100%）

### 変更ファイル
- `lib/graph.py` - `_compiled_graph`, `get_compiled_graph()`, `rewrite_query` スキップ条件, `stream_response_with_sources` 全面書き直し, `stream_response` の `build_rag_graph()` → `get_compiled_graph()`
- `tests/test_graph.py` - 上記テスト追加・更新
- `pyproject.toml` - pytest-cov 追加（開発依存）

## 2026-04-13 プロンプト優先順位改善 + 応答速度最適化 + ハイブリッド検索

### 作業内容
1. **プロンプト優先順位改善**: `RAG_SYSTEM_PROMPT` を3段階優先順位に変更、【ナレッジベース】/【一般知識】マーカー追加、拒否メッセージ削除
2. **応答速度最適化**: `get_compiled_graph()`, `create_llm()`, `_get_embeddings()` に `@lru_cache` 追加、`stream_response_with_sources()` を単一グラフ実行に書き直し
3. **ハイブリッド検索**: `search_relevant_documents()` に `use_hybrid` パラメータ追加（デフォルトTrue）、`match_documents_hybrid` SQL関数追加（ベクトル + キーワード全文検索の組み合わせスコア）

### テスト
- 全98テストパス、カバレッジ 91.36%
- `TestHybridSearch` 4テスト追加（hybrid_rpc呼び出し、query_text渡し、非hybrid時の既存RPC、combined_score）

### 変更ファイル
- `lib/rag_chain.py` - `use_hybrid` パラメータ追加、`_get_embeddings()` キャッシュ、プロンプト改善
- `lib/graph.py` - `get_compiled_graph()` キャッシュ、`stream_response_with_sources` 書き直し
- `lib/llm.py` - `create_llm()` キャッシュ
- `supabase-setup.sql` - `match_documents_hybrid` 関数 + GIN インデックス追加
- `tests/test_rag_chain.py` - `TestHybridSearch` + プロンプト関連テスト追加
- `tests/test_graph.py` - `TestGetCompiledGraph` + sources テスト更新
- `tests/conftest.py` - `clear_lru_caches` fixture 追加

### ユーザー対応事項
- ~~`supabase-setup.sql` の新しい部分（セクション10, 11）を Supabase SQL Editor で実行する必要あり~~ → Supabase MCP で適用済み

## 2026-04-13 match_documents_hybrid DB migration適用

### 作業内容
- Supabase MCP (`apply_migration`) を使い、DBに2つのmigrationを適用：
  1. `add_gin_index_for_fulltext_search`: `documents_content_tsvector_idx` GINインデックス作成
  2. `add_match_documents_hybrid_function`: `match_documents_hybrid` RPC関数作成
- `execute_sql` で関数存在を確認済み
- コード変更なし（DB migrationのみ）

### 状態
- ハイブリッド検索（ベクトル + キーワード全文検索）が本番DBで利用可能に
- `lib/rag_chain.py` の `search_relevant_documents(use_hybrid=True)` が正常動作する状態
