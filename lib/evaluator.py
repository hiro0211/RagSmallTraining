"""RAG evaluation: faithfulness, citation rate, and context relevance."""

import sys
from openai import OpenAI
from lib.rag_chain import search_relevant_documents, build_rag_prompt


def create_eval_dataset(
    questions: list[str], ground_truths: list[str]
) -> list[dict]:
    """Create evaluation dataset from questions and expected answers."""
    return [
        {"question": q, "ground_truth": gt}
        for q, gt in zip(questions, ground_truths)
    ]


def run_single_eval(question: str, ground_truth: str) -> dict:
    """Run RAG pipeline for a single question and collect eval data."""
    result = search_relevant_documents(question)
    messages = build_rag_prompt(question=question, context=result["context"])

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content

    contexts = [s.content for s in result["sources"]]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def _make_bigrams(text: str) -> set[str]:
    """Generate character bigram set after removing punctuation and whitespace."""
    import re

    cleaned = re.sub(r"[\s。、．，.,!！?？「」『』（）()・\-―～：:；;]", "", text)
    if len(cleaned) < 2:
        return set(cleaned)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def calc_faithfulness_simple(answer: str, context: str) -> float:
    """Calculate simple faithfulness score based on character bigram overlap.

    Measures what fraction of bigrams in the answer appear in the context.
    Higher score = more faithful (less hallucination).
    """
    if not answer or not context:
        return 0.0

    answer_bigrams = _make_bigrams(answer)
    context_bigrams = _make_bigrams(context)

    if not answer_bigrams:
        return 0.0

    overlap = answer_bigrams & context_bigrams
    return len(overlap) / len(answer_bigrams)


def calc_faithfulness_llm(answer: str, context: str) -> float:
    """Calculate faithfulness using LLM-as-judge."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはRAGシステムの回答品質を評価する審査員です。\n"
                    "回答がコンテキスト情報のみに基づいているか評価してください。\n"
                    "0.0（完全にハルシネーション）〜 1.0（完全に忠実）のスコアを数値のみで返してください。"
                ),
            },
            {
                "role": "user",
                "content": f"# コンテキスト:\n{context}\n\n# 回答:\n{answer}\n\nスコア:",
            },
        ],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0


def format_report(results: list[dict]) -> str:
    """Format evaluation results as a readable report."""
    lines = ["# RAG 評価レポート", ""]

    total_faithfulness = 0
    total_context_hit = 0

    for i, r in enumerate(results, 1):
        faithfulness = r.get("faithfulness", 0)
        context_hit = r.get("context_hit", False)
        total_faithfulness += faithfulness
        total_context_hit += 1 if context_hit else 0

        lines.append(f"## Q{i}: {r['question']}")
        lines.append(f"- **回答**: {r['answer'][:100]}...")
        lines.append(f"- **Faithfulness**: {faithfulness:.2f}")
        lines.append(f"- **Context Hit**: {'Yes' if context_hit else 'No'}")
        lines.append("")

    n = len(results)
    if n > 0:
        lines.append("---")
        lines.append(f"## 総合スコア")
        lines.append(f"- **平均 Faithfulness**: {total_faithfulness / n:.2f}")
        lines.append(f"- **ハルシネーション率**: {1 - total_faithfulness / n:.2f}")
        lines.append(f"- **Context Hit Rate**: {total_context_hit / n:.2f} ({total_context_hit}/{n})")

    return "\n".join(lines)


def run_evaluation(use_llm_judge: bool = False):
    """Run full evaluation with predefined test cases."""
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    test_cases = create_eval_dataset(
        questions=[
            "RAGとは何ですか？",
            "ベクトル検索の仕組みを教えて",
            "LangChainとは何ですか？",
            "今日の天気は？",
        ],
        ground_truths=[
            "RAGとは検索拡張生成（Retrieval-Augmented Generation）のことで、LLMの回答精度を向上させるために外部ナレッジベースから情報を検索してコンテキストとして提供する手法です。",
            "ベクトル検索はテキストを数値ベクトルに変換し、コサイン類似度などでベクトル間の距離を計算して関連データを検索する技術です。",
            "LangChainはLLMを活用したアプリケーション開発のためのフレームワークです。",
            "",
        ],
    )

    results = []
    for tc in test_cases:
        print(f"Evaluating: {tc['question']}")
        eval_data = run_single_eval(tc["question"], tc["ground_truth"])

        context = "\n".join(eval_data["contexts"])
        if use_llm_judge:
            faithfulness = calc_faithfulness_llm(eval_data["answer"], context)
        else:
            faithfulness = calc_faithfulness_simple(eval_data["answer"], context)

        eval_data["faithfulness"] = faithfulness
        eval_data["context_hit"] = len(eval_data["contexts"]) > 0
        results.append(eval_data)

    report = format_report(results)
    print("\n" + report)
    return results


if __name__ == "__main__":
    use_llm = "--llm-judge" in sys.argv
    run_evaluation(use_llm_judge=use_llm)
