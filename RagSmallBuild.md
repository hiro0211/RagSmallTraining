# RAGスモール版構築プロンプト — Claude Code用

## 概要

社内ナレッジ活用AIチャットボットのスモール版を、個人開発環境で構築するためのClaude Code実行プロンプト集です。
以下のフェーズ順に、Claude Codeへプロンプトを渡して実行してください。

---

## アーキテクチャ対応表

| 社内システム | スモール版 |
|---|---|
| Azure VM + nginx + Podman | Vercel (ホスティング) |
| Azure OpenAI (GPT-4o-mini) | OpenAI API (gpt-4o-mini) |
| Azure OpenAI (text-embedding-3-small) | OpenAI API (text-embedding-3-small) |
| Azure AI Search (Vectorstore) | Supabase pgvector |
| Streamlit (チャットUI) | Next.js App Router |
| LangChain (Python) | LangChain.js (TypeScript) |
| Celery + Redis | 不要（小規模のため） |
| Microsoft Entra ID (SSO) | 不要（個人利用のため） |
| Langfuse (監視) | 不要（開発段階のため） |

---

## 前提条件

- Node.js 20+
- pnpm（推奨）or npm
- OpenAI APIキー取得済み
- Supabase プロジェクト作成済み（無料枠）
- Supabase MCP サーバー接続済み（Claude Code or Claude Desktop）
- Vercelアカウント作成済み

---

## フェーズ0: プロジェクト初期化

### プロンプト

```
Next.jsプロジェクトを初期化してください。

条件:
- Next.js 15 App Router (TypeScript)
- パッケージマネージャー: pnpm
- Tailwind CSS v4
- src/ ディレクトリ構成
- 以下のパッケージをインストール:
  - langchain @langchain/openai @langchain/community
  - @supabase/supabase-js
  - ai (Vercel AI SDK)
- .env.local に以下の環境変数テンプレートを作成:
  OPENAI_API_KEY=
  SUPABASE_URL=
  SUPABASE_ANON_KEY=
  SUPABASE_SERVICE_ROLE_KEY=

プロジェクト名: RagSmallTraining
```

---

## フェーズ1: Supabase pgvector セットアップ

### プロンプト（Supabase MCP経由で実行）

```
Supabase MCPを使って、RAG用のベクトルデータベースをセットアップしてください。

1. pgvector拡張を有効化:
   CREATE EXTENSION IF NOT EXISTS vector;

2. ドキュメント格納テーブルを作成:
   テーブル名: documents
   カラム:
   - id: bigint (自動生成, 主キー)
   - content: text (チャンクのテキスト内容)
   - metadata: jsonb (ソースファイル名、ページ番号など)
   - embedding: vector(1536) (text-embedding-3-smallの次元数)
   - created_at: timestamptz (デフォルト: now())

3. ベクトル類似検索用のRPC関数を作成:
   関数名: match_documents
   引数:
   - query_embedding: vector(1536)
   - match_threshold: float (デフォルト: 0.7)
   - match_count: int (デフォルト: 5)
   戻り値: id, content, metadata, similarity
   ロジック: コサイン類似度で降順ソートし、閾値以上のものをmatch_count件返す

4. embeddingカラムにHNSWインデックスを作成:
   CREATE INDEX ON documents
   USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);

各SQLの実行結果を確認してください。
```

---

## フェーズ2: Embeddingパイプライン構築

### プロンプト

```
ドキュメントをチャンク分割 → Embedding → Supabaseに格納するスクリプトを作成してください。

ファイル: src/lib/embedding-pipeline.ts

処理フロー:
1. ドキュメント読み込み
   - PDFファイル: LangChainのPDFLoaderを使用
   - テキストファイル: TextLoaderを使用
   - 入力ディレクトリ: docs/ フォルダ

2. チャンク分割
   - LangChainのRecursiveCharacterTextSplitterを使用
   - チャンクサイズ: 1000文字
   - オーバーラップ: 200文字
   - セパレーター: ["\n\n", "\n", "。", ".", " "]（日本語対応）

3. Embedding生成
   - モデル: text-embedding-3-small (OpenAI)
   - LangChainのOpenAIEmbeddingsを使用
   - バッチ処理（一度に100チャンクずつ）

4. Supabaseへ格納
   - @supabase/supabase-js を使って documents テーブルに INSERT
   - metadataにはソースファイル名とチャンクインデックスを含める

5. CLIから実行可能にする:
   pnpm run embed -- --dir ./docs

実行用のpackage.jsonスクリプトも追加してください:
"embed": "npx tsx src/lib/embedding-pipeline.ts"
```

---

## フェーズ3: RAG検索ロジック

### プロンプト

```
RAGの検索→プロンプト拡張→LLM回答生成のロジックを作成してください。

ファイル: src/lib/rag-chain.ts

処理フロー:
1. ユーザー質問をEmbedding化
   - text-embedding-3-small を使用

2. Supabaseで類似検索
   - match_documents RPC関数を呼び出し
   - top_k: 5件
   - similarity閾値: 0.7

3. プロンプトテンプレート（LangChain PromptTemplate使用）:

   あなたは社内ナレッジに基づいて質問に回答するアシスタントです。
   以下のコンテキスト情報を参考に、質問に正確に回答してください。
   コンテキストに情報がない場合は、「この情報はナレッジベースに含まれていません」と回答してください。

   # コンテキスト:
   {context}

   # 質問:
   {question}

   # 回答:

4. LLM呼び出し
   - モデル: gpt-4o-mini
   - LangChainのChatOpenAIを使用
   - temperature: 0（正確性重視）
   - streaming対応

5. エクスポート:
   - queryRAG(question: string) → AsyncGenerator<string> 形式
   - 検索結果のソース情報も返せるようにする
```

---

## フェーズ4: チャットUI & APIルート

### プロンプト

```
Next.js App RouterでチャットUIとAPIルートを作成してください。

### APIルート
ファイル: src/app/api/chat/route.ts

- POSTエンドポイント
- Vercel AI SDK の streamText を活用
- リクエストボディ: { message: string }
- フェーズ3で作成した queryRAG を呼び出し
- ストリーミングレスポンスで返す
- エラーハンドリング付き

### チャットUI
ファイル: src/app/page.tsx

- Vercel AI SDK の useChat フックを使用
- シンプルなチャットインターフェース:
  - メッセージ一覧（ユーザー/アシスタント表示分け）
  - テキスト入力フィールド
  - 送信ボタン
  - ストリーミング中のローディング表示
- Tailwind CSSでスタイリング
- レスポンシブ対応
- ダークモード対応（prefers-color-scheme）

デザイン要件:
- 画面中央にチャット領域
- 最大幅 800px
- メッセージはバブル形式（ユーザー: 右寄せ青系、AI: 左寄せグレー系）
- 入力欄は画面下部に固定
```

---

## フェーズ5: 動作テスト

### プロンプト

```
RAGシステムの動作テストを実施してください。

1. テスト用ドキュメントの準備:
   docs/test.md に以下のサンプルコンテンツを作成:
   - RAGの技術概要（500文字程度）
   - ベクトル検索の仕組み（500文字程度）
   - LangChainの基本概念（500文字程度）

2. Embeddingパイプラインの実行:
   pnpm run embed -- --dir ./docs
   - Supabaseのdocumentsテーブルにレコードが格納されたことを確認

3. 開発サーバー起動:
   pnpm dev

4. テストケース:
   - 質問1: 「RAGとは何ですか？」 → コンテキストに基づいた回答が返ること
   - 質問2: 「ベクトル検索の仕組みを教えて」 → 類似検索が機能すること
   - 質問3: 「今日の天気は？」 → ナレッジベースにない旨の回答が返ること

各ステップの実行結果とエラーがあれば報告してください。
```

---

## フェーズ6: Vercelデプロイ

### プロンプト

```
Vercelにデプロイしてください。

1. vercel.json の作成（必要に応じて）
2. 環境変数の設定確認:
   - OPENAI_API_KEY
   - SUPABASE_URL
   - SUPABASE_ANON_KEY
   - SUPABASE_SERVICE_ROLE_KEY
3. vercel --prod でデプロイ実行
4. デプロイURLでの動作確認

注意: SUPABASE_SERVICE_ROLE_KEY はサーバーサイド（APIルート）でのみ使用し、
クライアントサイドには絶対に露出させないこと。
```

---

## ディレクトリ構成（完成イメージ）

```
RagSmallTraining/
├── docs/                          # RAG対象ドキュメント
│   └── test.md
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── chat/
│   │   │       └── route.ts       # チャットAPIエンドポイント
│   │   ├── layout.tsx
│   │   └── page.tsx               # チャットUI
│   └── lib/
│       ├── embedding-pipeline.ts  # Embedding生成スクリプト
│       ├── rag-chain.ts           # RAG検索・回答生成ロジック
│       └── supabase.ts            # Supabaseクライアント初期化
├── .env.local
├── package.json
├── tsconfig.json
└── next.config.ts
```

---

## 社内RAGシステムとの対応（学習ポイント）

| 社内システムの処理 | スモール版での体験 |
|---|---|
| PDF → Document Loader → テキスト抽出 | LangChain PDFLoader でPDF読み込み |
| Excel → Markdown変換 | テキストファイルのチャンク分割で代替 |
| チャンキング（意味単位分割） | RecursiveCharacterTextSplitter で分割戦略を設計 |
| Embeddingモデルでベクトル化 | text-embedding-3-small でベクトル生成 |
| Azure AI Searchにインデックス格納 | Supabase pgvector にINSERT + HNSWインデックス |
| 質問ベクトル化 → コサイン類似度検索 | match_documents RPC関数で類似検索 |
| システムプロンプト + コンテキスト埋め込み | LangChain PromptTemplate でプロンプト拡張 |
| Azure OpenAI で回答生成 | OpenAI API (gpt-4o-mini) で回答生成 |
| Streamlit でチャット表示 | Next.js + Vercel AI SDK でチャットUI |

---

## 発展課題（余裕があれば）

- **ハイブリッド検索**: pgvectorのベクトル検索 + PostgreSQL全文検索の組み合わせ
- **リランキング**: 検索結果をLLMで再スコアリングして精度向上
- **チャット履歴**: 会話コンテキストを保持して多ターン対話に対応
- **ソース表示**: 回答に使用したドキュメントチャンクの参照元を表示
- **Langfuse統合**: LLM呼び出しのトレーシング・コスト監視を追加
- **認証**: NextAuth.js でログイン機能を追加