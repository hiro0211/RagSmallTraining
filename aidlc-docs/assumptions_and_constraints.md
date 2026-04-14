# 前提条件と制約

## 決めてよいこと（開発チームで判断可能）

- コード内の関数名・変数名・クラス名（英語）
- LangGraph ノードの追加順序と内部実装
- テストのモック戦略
- キャッシュ戦略（`lru_cache` の適用箇所）
- バッチサイズ（Embedding生成、Supabase INSERT）
- ログ出力のフォーマット

## 決めてはいけないこと（人間の承認が必要）

- Reranking モデルの変更（ローカル → Cohere API 等）
- Embedding モデルの変更（`text-embedding-3-small` → 他モデル）
- Supabase テーブルスキーマの変更（本番環境への影響）
- メタデータ項目の追加・変更（ビジネス要件に依存）
- RAGAS 評価の合格基準値（ビジネス判断）
- 検索パイプラインの重み配分（ベクトル/キーワード比率）の変更

## 現時点の制約

- 技術的制約:
  - Embedding モデルは `text-embedding-3-small`（1536次元）で統一（変更時は全チャンク再インデックス必要）
  - Cross-Encoder はCPU実行（GPU環境なし）
  - Supabase の RPC 関数は SQL で定義（複雑なロジックの制約あり）
  - `langchain_experimental` は API が不安定な可能性あり
- ビジネス的制約:
  - MVP段階のため、精度改善の定量目標は未設定
  - テストデータセットのドメイン知識が限定的

## 保留論点

- Cross-Encoder のGPU対応（将来的にAWS上でのデプロイ時）
- Embedding モデルのファインチューニング（Lv.5 の範囲）
- マルチテナント対応時のメタデータフィルタリング設計
- HyDE / Multi-Query の動的切り替えロジック（質問の複雑度に応じた分岐）
