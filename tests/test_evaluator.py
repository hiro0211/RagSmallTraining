"""lib/evaluator.py のテスト — RAGAS 4指標 + テストデータセット."""

from unittest.mock import patch, MagicMock
import pytest


class TestEvalDataset:
    def test_create_eval_dataset(self):
        from lib.evaluator import create_eval_dataset

        dataset = create_eval_dataset(
            questions=["RAGとは？", "天気は？"],
            ground_truths=["検索拡張生成です。", ""],
        )
        assert len(dataset) == 2
        assert dataset[0].question == "RAGとは？"
        assert dataset[0].ground_truth == "検索拡張生成です。"

    def test_default_test_cases_has_10_questions(self):
        from lib.evaluator import get_default_test_cases

        cases = get_default_test_cases()
        assert len(cases) == 10
        assert all(c.question for c in cases)
        assert all(c.ground_truth for c in cases)


class TestRunSingleEval:
    def test_populates_answer_and_contexts(self):
        from lib.evaluator import EvalCase, run_single_eval

        with (
            patch("lib.evaluator.search_relevant_documents") as mock_search,
            patch("lib.evaluator.format_sources_as_context") as mock_format,
            patch("lib.evaluator.OpenAI") as MockOpenAI,
        ):
            mock_src = MagicMock()
            mock_src.content = "RAGは検索拡張生成です。"
            mock_search.return_value = {"sources": [mock_src]}
            mock_format.return_value = "コンテキスト"

            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "RAGは検索拡張生成のことです。"
            mock_client.chat.completions.create.return_value = mock_response

            case = EvalCase(question="RAGとは？", ground_truth="検索拡張生成です。")
            result = run_single_eval(case)

            assert result.answer == "RAGは検索拡張生成のことです。"
            assert len(result.contexts) == 1


class TestCalcFaithfulnessSimple:
    def test_identical_text_returns_1(self):
        from lib.evaluator import calc_faithfulness_simple

        text = "RAGとは検索拡張生成のことです。"
        assert calc_faithfulness_simple(text, text) == 1.0

    def test_faithful_answer_scores_high(self):
        from lib.evaluator import calc_faithfulness_simple

        context = "RAGとは検索拡張生成（Retrieval-Augmented Generation）のことです。"
        answer = "RAGとは検索拡張生成のことです。"
        score = calc_faithfulness_simple(answer, context)
        assert score > 0.5

    def test_unrelated_text_scores_low(self):
        from lib.evaluator import calc_faithfulness_simple

        context = "RAGとは検索拡張生成のことです。"
        answer = "今日は晴れで気温は25度になるでしょう。"
        score = calc_faithfulness_simple(answer, context)
        assert score < 0.3

    def test_empty_input_returns_0(self):
        from lib.evaluator import calc_faithfulness_simple

        assert calc_faithfulness_simple("", "context") == 0.0
        assert calc_faithfulness_simple("answer", "") == 0.0


class TestRagasMetrics:
    """RAGAS 4指標の LLM-as-Judge テスト（モック使用）."""

    def _mock_llm_judge(self, score_str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = score_str
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_context_recall(self):
        from lib.evaluator import calc_context_recall

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_llm_judge("0.85")
            score = calc_context_recall(["RAGの説明"], "RAGは検索拡張生成です。")
            assert score == 0.85

    def test_context_precision(self):
        from lib.evaluator import calc_context_precision

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_llm_judge("0.90")
            score = calc_context_precision(["RAGの説明"], "RAGは検索拡張生成です。")
            assert score == 0.90

    def test_faithfulness(self):
        from lib.evaluator import calc_faithfulness

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_llm_judge("0.95")
            score = calc_faithfulness("RAGは検索拡張生成です。", ["RAGの説明"])
            assert score == 0.95

    def test_answer_relevancy(self):
        from lib.evaluator import calc_answer_relevancy

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_llm_judge("0.80")
            score = calc_answer_relevancy("RAGとは？", "RAGは検索拡張生成です。")
            assert score == 0.80

    def test_invalid_llm_response_returns_0(self):
        from lib.evaluator import calc_context_recall

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_llm_judge("スコアは高いです")
            score = calc_context_recall(["ctx"], "gt")
            assert score == 0.0


class TestEvaluateCase:
    def test_returns_all_metrics(self):
        from lib.evaluator import EvalCase, evaluate_case

        case = EvalCase(
            question="RAGとは？",
            ground_truth="検索拡張生成です。",
            answer="RAGは検索拡張生成のことです。",
            contexts=["RAGの説明"],
        )

        with patch("lib.evaluator.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "0.85"
            mock_client.chat.completions.create.return_value = mock_response

            result = evaluate_case(case, use_llm_judge=True)

        assert "context_recall" in result
        assert "context_precision" in result
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "faithfulness_simple" in result
        assert "context_hit" in result
        assert result["context_hit"] is True

    def test_skips_llm_judge_when_disabled(self):
        from lib.evaluator import EvalCase, evaluate_case

        case = EvalCase(
            question="RAGとは何ですか",
            ground_truth="RAGは検索拡張生成のことです",
            answer="RAGは検索拡張生成のことです。外部ナレッジベースから情報を検索してLLMに提供する手法です。",
            contexts=["RAGは検索拡張生成のことです。外部ナレッジベースから情報を検索してLLMに提供する手法です。"],
        )

        result = evaluate_case(case, use_llm_judge=False)

        assert result["context_recall"] == 0.0
        assert result["faithfulness_simple"] > 0


class TestFormatReport:
    def test_includes_all_metrics(self):
        from lib.evaluator import format_report

        results = [
            {
                "question": "RAGとは？",
                "answer": "検索拡張生成です。",
                "contexts_count": 3,
                "context_recall": 0.85,
                "context_precision": 0.90,
                "faithfulness": 0.95,
                "answer_relevancy": 0.80,
                "faithfulness_simple": 0.70,
                "context_hit": True,
            },
        ]
        report = format_report(results)

        assert "Context Recall" in report
        assert "Context Precision" in report
        assert "Faithfulness" in report
        assert "Answer Relevancy" in report
        assert "総合スコア" in report
