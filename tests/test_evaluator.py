"""Tests for RAG evaluator."""

from unittest.mock import patch, MagicMock
import pytest


class TestEvalDataset:
    def test_create_eval_dataset(self):
        from lib.evaluator import create_eval_dataset

        questions = ["RAGとは？", "今日の天気は？"]
        ground_truths = [
            "RAGとは検索拡張生成のことです。",
            "",
        ]
        dataset = create_eval_dataset(questions, ground_truths)
        assert len(dataset) == 2
        assert dataset[0]["question"] == "RAGとは？"
        assert dataset[0]["ground_truth"] == "RAGとは検索拡張生成のことです。"


class TestRunEvaluation:
    def test_returns_metrics(self):
        with (
            patch("lib.evaluator.search_relevant_documents") as mock_search,
            patch("lib.evaluator.OpenAI") as MockOpenAI,
        ):
            mock_search.return_value = {
                "context": "RAGは検索拡張生成です。",
                "sources": [MagicMock(content="RAGは検索拡張生成です。")],
            }
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "RAGは検索拡張生成のことです。"
            mock_client.chat.completions.create.return_value = mock_response

            from lib.evaluator import run_single_eval

            result = run_single_eval(
                question="RAGとは？",
                ground_truth="RAGは検索拡張生成のことです。",
            )
            assert "question" in result
            assert "answer" in result
            assert "contexts" in result
            assert "ground_truth" in result


class TestCalcFaithfulness:
    def test_faithful_answer(self):
        from lib.evaluator import calc_faithfulness_simple

        context = "RAGとは検索拡張生成（Retrieval-Augmented Generation）のことです。"
        answer = "RAGとは検索拡張生成のことです。"
        score = calc_faithfulness_simple(answer, context)
        assert 0 <= score <= 1
        assert score > 0.5  # High faithfulness

    def test_hallucinated_answer(self):
        from lib.evaluator import calc_faithfulness_simple

        context = "RAGとは検索拡張生成のことです。"
        answer = "RAGは2025年にGoogleが開発した技術で、量子コンピュータと連携して動作します。"
        score = calc_faithfulness_simple(answer, context)
        assert 0 <= score <= 1
        assert score < 0.5  # Low faithfulness


class TestBigramFaithfulness:
    def test_identical_text_returns_1(self):
        from lib.evaluator import calc_faithfulness_simple

        text = "RAGとは検索拡張生成のことです。"
        score = calc_faithfulness_simple(text, text)
        assert score == 1.0

    def test_unrelated_text_returns_low(self):
        from lib.evaluator import calc_faithfulness_simple

        context = "RAGとは検索拡張生成のことです。"
        answer = "今日は晴れで気温は25度になるでしょう。"
        score = calc_faithfulness_simple(answer, context)
        assert score < 0.3


class TestFormatReport:
    def test_format_report(self):
        from lib.evaluator import format_report

        results = [
            {
                "question": "RAGとは？",
                "answer": "検索拡張生成です。",
                "contexts": ["RAGは検索拡張生成です。"],
                "ground_truth": "RAGは検索拡張生成のことです。",
                "faithfulness": 0.9,
                "context_hit": True,
            },
        ]
        report = format_report(results)
        assert "RAGとは？" in report
        assert "faithfulness" in report.lower() or "Faithfulness" in report
