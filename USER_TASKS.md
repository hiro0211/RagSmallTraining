# ユーザーが対応すべきこと

## 1. OpenAI API キー取得
1. https://platform.openai.com にアクセス（アカウントがなければ作成）
2. 左メニュー「API keys」→「Create new secret key」
3. キーをコピーして安全な場所に保存
4. 「Settings」→「Billing」でクレジットを追加（$5〜で十分）

## 2. Supabase プロジェクト作成
1. https://supabase.com にアクセス（GitHub アカウントでサインアップ可）
2. 「New Project」をクリック
3. プロジェクト名: `RagSmallTraining`（任意）、リージョン: `Northeast Asia (Tokyo)` 推奨
4. データベースパスワードを設定して「Create new project」
5. プロジェクト作成完了後、以下をメモ:
   - **Project URL:** Settings → API → Project URL
   - **Publishable key:** Settings → API Keys → Publishable key（`sb_publishable_...`）
   - **Secret key:** Settings → API Keys → Secret key（`sb_secret_...`）

## 3. Supabase SQL 実行
`supabase-setup.sql` の内容を Supabase SQL Editor で実行:
1. Supabase ダッシュボード → 左メニュー「SQL Editor」
2. `supabase-setup.sql` の内容を貼り付けて「Run」

## 4. .env.local に値を記入
```
OPENAI_API_KEY=sk-xxxx
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_xxxx
SUPABASE_SECRET_KEY=sb_secret_xxxx
```

## 5. Embedding 実行 & Streamlit 起動
```bash
# ドキュメントをベクトル化
uv run python lib/embedding_pipeline.py --dir ./docs

# Streamlit 起動
uv run streamlit run app.py
```

## テストケース
- 「RAGとは何ですか？」→ コンテキストに基づいた回答
- 「ベクトル検索の仕組みを教えて」→ 類似検索が機能
- 「今日の天気は？」→ 「ナレッジベースに含まれていません」
