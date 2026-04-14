"""RAG評価: RAGAS 4指標 + 簡易評価の統合モジュール."""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI
from lib.rag_chain import (
    search_relevant_documents,
    format_sources_as_context,
    build_rag_prompt,
)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """評価用の1ケース（質問 + 正解 + RAG実行結果）."""

    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)


@dataclass
class EvalScores:
    """RAGAS 4指標 + 簡易指標のスコア."""

    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    # 簡易指標（LLM不要）
    context_hit: bool = False
    faithfulness_simple: float = 0.0


# ---------------------------------------------------------------------------
# テストデータセット
# ---------------------------------------------------------------------------

def create_eval_dataset(
    questions: list[str], ground_truths: list[str]
) -> list[EvalCase]:
    """質問と正解回答のペアから評価データセットを作成."""
    return [
        EvalCase(question=q, ground_truth=gt)
        for q, gt in zip(questions, ground_truths)
    ]


def get_default_test_cases() -> list[EvalCase]:
    """docs/test.md ベースのデフォルトテストケースを返す."""
    return create_eval_dataset(
        questions=[
            "RAGとは何ですか？",
            "ベクトル検索の仕組みを教えてください",
            "ハイブリッド検索のメリットは何ですか？",
            "セマンティックチャンキングとは何ですか？",
            "Rerankingの仕組みを説明してください",
            "HyDEとは何ですか？",
            "Multi-Query Expansionの目的は何ですか？",
            "RAGASの4つの評価指標を教えてください",
            "メタデータフィルタリングの利点は何ですか？",
            "Cross-EncoderとBi-Encoderの違いは何ですか？",
        ],
        ground_truths=[
            "RAGとは検索拡張生成（Retrieval-Augmented Generation）のことで、LLMの回答精度を向上させるために外部のナレッジベースから関連情報を検索し、その情報をコンテキストとしてLLMに提供する手法です。",
            "ベクトル検索はテキストを数値ベクトル（Embedding）に変換し、ベクトル間のコサイン類似度を計算して関連するデータを検索する技術です。OpenAIのtext-embedding-3-smallなどのモデルで1536次元のベクトルに変換します。",
            "ハイブリッド検索はベクトル検索とキーワード検索を組み合わせた手法で、ベクトル検索の弱点（固有名詞の完全一致を保証しない）とキーワード検索の弱点（言い換えや類義語を見逃す）を相互補完し、単独の検索方式よりも高い精度を実現します。",
            "セマンティックチャンキングは文間の意味的類似度を計算し、類似度が大きく低下する箇所で分割する手法です。固定長分割と異なり、内容の一貫性が保たれたチャンクが生成されます。",
            "Rerankingは初回検索で取得した候補ドキュメントをCross-Encoderで再スコアリングして順位を並べ替える手法です。Bi-Encoderで広く候補を取得し、Cross-Encoderで精密に上位を絞り込む2段階パイプラインが一般的です。",
            "HyDEはHypothetical Document Embeddingsの略で、ユーザーの質問に対してLLMが仮説的な回答を生成し、その回答のEmbeddingで検索を行う手法です。短い質問よりも詳細な仮説回答のほうが実際のドキュメントに近いEmbeddingになります。",
            "Multi-Query Expansionは1つの質問をLLMで3〜5の異なる言い換えに展開し、それぞれで並列に検索を行い結果を統合する手法です。表現の違いによる検索漏れを防ぎ、Recall（検索の網羅性）を向上させます。",
            "RAGASの4指標はContext Recall（検索結果に正解情報が含まれているか）、Context Precision（検索結果の上位に正解情報があるか）、Faithfulness（回答がコンテキストに忠実か）、Answer Relevancy（回答が質問に関連しているか）です。",
            "メタデータフィルタリングはベクトル検索の前段階でメタデータ条件に基づいてドキュメントを絞り込む手法です。検索対象を事前に限定することで、不要なチャンクがヒットすることを防ぎ、検索精度と速度の両方を向上させます。",
            "Bi-Encoderはクエリとドキュメントをそれぞれ独立にベクトル化して距離で類似度を計算するため高速ですが精度が劣ります。Cross-Encoderはクエリとドキュメントのペアを同時に入力して直接関連度スコアを出力するため精度が高いですが低速です。",
        ],
    )


# ---------------------------------------------------------------------------
# RAGパイプライン実行
# ---------------------------------------------------------------------------

def run_single_eval(case: EvalCase) -> EvalCase:
    """1つの質問に対してRAGパイプラインを実行し、回答とコンテキストを取得."""
    result = search_relevant_documents(case.question)
    context = format_sources_as_context(result["sources"])
    messages = build_rag_prompt(question=case.question, context=context)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    case.answer = response.choices[0].message.content or ""
    case.contexts = [s.content for s in result["sources"]]
    return case


# ---------------------------------------------------------------------------
# 簡易評価（LLM不要）
# ---------------------------------------------------------------------------

def _make_bigrams(text: str) -> set[str]:
    """句読点・空白を除去した文字バイグラムの集合を返す."""
    import re
    cleaned = re.sub(r"[\s。、．，.,!！?？「」『』（）()・\-―～：:；;]", "", text)
    if len(cleaned) < 2:
        return set(cleaned)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def calc_faithfulness_simple(answer: str, context: str) -> float:
    """文字バイグラムの重複率で簡易的な忠実度を計算."""
    if not answer or not context:
        return 0.0
    answer_bigrams = _make_bigrams(answer)
    context_bigrams = _make_bigrams(context)
    if not answer_bigrams:
        return 0.0
    overlap = answer_bigrams & context_bigrams
    return len(overlap) / len(answer_bigrams)


# ---------------------------------------------------------------------------
# RAGAS 4指標（LLM-as-Judge）
# ---------------------------------------------------------------------------

def _llm_judge(system_prompt: str, user_prompt: str) -> float:
    """LLMに0.0〜1.0のスコアを返させる共通関数."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except (ValueError, AttributeError):
        return 0.0


def calc_context_recall(contexts: list[str], ground_truth: str) -> float:
    """Context Recall: 正解回答の情報が検索結果に含まれているかを計測."""
    context_text = "\n".join(contexts)
    return _llm_judge(
        system_prompt=(
            "あなたはRAGシステムの検索品質を評価する審査員です。\n"
            "正解回答（Ground Truth）の各文が、検索されたコンテキストのいずれかに帰属できるかを判定してください。\n"
            "帰属できる文の割合を 0.0〜1.0 のスコアで返してください。数値のみ返してください。"
        ),
        user_prompt=(
            f"# 検索されたコンテキスト:\n{context_text}\n\n"
            f"# 正解回答（Ground Truth）:\n{ground_truth}\n\nスコア:"
        ),
    )


def calc_context_precision(contexts: list[str], ground_truth: str) -> float:
    """Context Precision: 検索結果の上位に正解関連情報が集中しているかを計測."""
    context_with_rank = "\n".join(
        f"[順位{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    return _llm_judge(
        system_prompt=(
            "あなたはRAGシステムの検索品質を評価する審査員です。\n"
            "検索結果の各チャンクが正解回答に関連しているかを判定し、\n"
            "関連するチャンクが上位に集中しているほど高いスコアを付けてください。\n"
            "0.0〜1.0 のスコアを数値のみで返してください。"
        ),
        user_prompt=(
            f"# 検索結果（順位付き）:\n{context_with_rank}\n\n"
            f"# 正解回答（Ground Truth）:\n{ground_truth}\n\nスコア:"
        ),
    )


def calc_faithfulness(answer: str, contexts: list[str]) -> float:
    """Faithfulness: 回答がコンテキストに忠実かを計測."""
    context_text = "\n".join(contexts)
    return _llm_judge(
        system_prompt=(
            "あなたはRAGシステムの回答品質を評価する審査員です。\n"
            "回答に含まれる各主張（claim）が、コンテキストから裏付けられるかを判定してください。\n"
            "裏付けられる主張の割合を 0.0（完全にハルシネーション）〜 1.0（完全に忠実）のスコアで返してください。\n"
            "数値のみ返してください。"
        ),
        user_prompt=(
            f"# コンテキスト:\n{context_text}\n\n"
            f"# 回答:\n{answer}\n\nスコア:"
        ),
    )


def calc_answer_relevancy(question: str, answer: str) -> float:
    """Answer Relevancy: 回答が質問に関連しているかを計測."""
    return _llm_judge(
        system_prompt=(
            "あなたはRAGシステムの回答品質を評価する審査員です。\n"
            "回答がユーザーの質問にどれだけ関連しているかを評価してください。\n"
            "質問の意図に正確に答えているほど高いスコアを付けてください。\n"
            "0.0（完全に無関係）〜 1.0（完全に関連）のスコアを数値のみで返してください。"
        ),
        user_prompt=(
            f"# 質問:\n{question}\n\n"
            f"# 回答:\n{answer}\n\nスコア:"
        ),
    )


# ---------------------------------------------------------------------------
# 統合評価
# ---------------------------------------------------------------------------

def evaluate_case(case: EvalCase, use_llm_judge: bool = True) -> dict:
    """1ケースに対して全指標を計算し、結果を辞書で返す."""
    context_text = "\n".join(case.contexts)

    scores = EvalScores(
        context_hit=len(case.contexts) > 0,
        faithfulness_simple=calc_faithfulness_simple(case.answer, context_text),
    )

    if use_llm_judge:
        scores.context_recall = calc_context_recall(
            case.contexts, case.ground_truth
        )
        scores.context_precision = calc_context_precision(
            case.contexts, case.ground_truth
        )
        scores.faithfulness = calc_faithfulness(case.answer, case.contexts)
        scores.answer_relevancy = calc_answer_relevancy(
            case.question, case.answer
        )

    return {
        "question": case.question,
        "ground_truth": case.ground_truth,
        "answer": case.answer,
        "contexts_count": len(case.contexts),
        **asdict(scores),
    }


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------

def format_report(results: list[dict]) -> str:
    """評価結果を日本語のMarkdownレポートに整形."""
    lines = ["# RAG 評価レポート（RAGAS 4指標）", ""]

    # 集計用
    totals = {
        "context_recall": 0.0,
        "context_precision": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "faithfulness_simple": 0.0,
        "context_hit": 0,
    }

    for i, r in enumerate(results, 1):
        lines.append(f"## Q{i}: {r['question']}")
        lines.append(f"- 回答: {r['answer'][:150]}...")
        lines.append(f"- 検索ヒット数: {r['contexts_count']}")
        lines.append(f"- Context Recall: {r['context_recall']:.2f}")
        lines.append(f"- Context Precision: {r['context_precision']:.2f}")
        lines.append(f"- Faithfulness: {r['faithfulness']:.2f}")
        lines.append(f"- Answer Relevancy: {r['answer_relevancy']:.2f}")
        lines.append(f"- Faithfulness（簡易）: {r['faithfulness_simple']:.2f}")
        lines.append("")

        for key in totals:
            if key == "context_hit":
                totals[key] += 1 if r[key] else 0
            else:
                totals[key] += r[key]

    n = len(results)
    if n > 0:
        lines.append("---")
        lines.append("## 総合スコア")
        lines.append("")
        lines.append("| 指標 | 平均スコア |")
        lines.append("|------|-----------|")
        lines.append(f"| Context Recall | {totals['context_recall'] / n:.2f} |")
        lines.append(f"| Context Precision | {totals['context_precision'] / n:.2f} |")
        lines.append(f"| Faithfulness | {totals['faithfulness'] / n:.2f} |")
        lines.append(f"| Answer Relevancy | {totals['answer_relevancy'] / n:.2f} |")
        lines.append(f"| Faithfulness（簡易） | {totals['faithfulness_simple'] / n:.2f} |")
        lines.append(f"| Context Hit Rate | {totals['context_hit']}/{n} ({totals['context_hit'] / n:.2f}) |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン実行
# ---------------------------------------------------------------------------

def run_evaluation(
    use_llm_judge: bool = True,
    save_report: bool = True,
    save_json: bool = True,
) -> list[dict]:
    """全テストケースで評価を実行し、レポートを出力."""
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    test_cases = get_default_test_cases()
    results = []

    for case in test_cases:
        print(f"評価中: {case.question}")
        run_single_eval(case)
        result = evaluate_case(case, use_llm_judge=use_llm_judge)
        results.append(result)

    report = format_report(results)
    print("\n" + report)

    if save_report:
        report_path = Path("aidlc-docs/ragas_baseline.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nレポート保存先: {report_path}")

    if save_json:
        json_path = Path("aidlc-docs/ragas_baseline.json")
        json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON保存先: {json_path}")

    return results


if __name__ == "__main__":
    use_llm = "--no-llm" not in sys.argv
    run_evaluation(use_llm_judge=use_llm)
