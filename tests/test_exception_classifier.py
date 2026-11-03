from pathlib import Path
import pandas as pd
import pytest

from exception_classifier import (
    LABELS, train_exception_classifier, predict_texts,
    classify_with_context, evaluate_classifier, split_sentences,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return train_exception_classifier(ROOT / "data" / "exception_training.csv")


def test_training_uses_both_positive_and_negative_examples(model):
    assert model.training_rows > model.positive_training_rows > 0
    assert model.validation_rows > 0
    assert set(model.binarizer.classes_) == set(LABELS)


def test_clear_trade_break_is_detected(model):
    result = predict_texts(model, ["Quantity mismatch: internal 100 shares but broker reports 90."])[0]
    assert result["exception_detected"] is True
    assert "QUANTITY" in result["suggested_labels"]


def test_clean_reconciled_sentence_is_not_flagged(model):
    result = predict_texts(model, ["All fields agree across both systems; no discrepancy or duplicate remains."])[0]
    assert result["exception_detected"] is False


def test_multi_label_output_can_return_more_than_one_category(model):
    result = predict_texts(model, ["The broker reports 90 shares instead of 100 and execution price differs from USD 40.90."])[0]
    assert result["exception_detected"] is True
    assert "QUANTITY" in result["suggested_labels"]
    assert "PRICE_CURRENCY" in result["suggested_labels"]


def test_context_window_preserves_sentence_offsets(model):
    note = "The internal record says 100 shares. The broker confirmation says 90 shares."
    result = classify_with_context(model, note, context_radius=1)
    assert len(result) == 2
    assert result.iloc[1]["sentence"] == "The broker confirmation says 90 shares."
    assert note[int(result.iloc[1]["start_offset"]):int(result.iloc[1]["end_offset"])] == result.iloc[1]["sentence"]
    assert "internal record says 100 shares" in result.iloc[1]["context_text"]


def test_sentence_split_does_not_split_decimal_prices():
    parts = split_sentences("Internal price is USD 25.10. Broker price is USD 25.20.")
    assert len(parts) == 2
    assert "25.10" in parts[0]["sentence"]
    assert "25.20" in parts[1]["sentence"]


def test_evaluation_reports_detection_and_category_metrics(model):
    predictions, summary = evaluate_classifier(model, ROOT / "data" / "exception_evaluation.csv")
    assert len(predictions) == summary["evaluation_rows"]
    assert 0 <= summary["detection_f1"] <= 1
    assert 0 <= summary["end_to_end_category_macro_f1"] <= 1
    assert set(summary["per_label_end_to_end"]) == set(LABELS)
    assert "not a production benchmark" in summary["note"].lower()


def test_candidate_margin_is_not_reported_as_probability(model):
    result = predict_texts(model, ["Price difference: broker execution price is inconsistent."])[0]
    assert "detection_margin" in result
    assert "confidence_probability" not in result
    assert all("margin" in item for item in result["category_suggestions"])


def test_clean_unique_counterpart_statement_suppresses_duplicate_false_positive(model):
    result = predict_texts(model, ["The custodian confirmation is present and linked to one unique internal transaction."])[0]
    assert result["exception_detected"] is False
    assert result["raw_detector_flag"] in {True, False}
    assert "DUPLICATE_OR_MISSING_LINK" in result["postprocess_suppressed_labels"]


def test_explicit_duplicate_statement_remains_a_suggestion(model):
    result = predict_texts(model, ["Duplicate booking detected: the same transaction appears twice in the external feed."])[0]
    assert result["exception_detected"] is True
    assert "DUPLICATE_OR_MISSING_LINK" in result["suggested_labels"]
