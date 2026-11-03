"""Run reproducible synthetic extraction, exception-classifier and trade checks."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_extraction import ANSWERABLE, extract_fields
from exception_classifier import (
    LABELS, train_exception_classifier, evaluate_classifier,
    classify_with_context,
)
from reconciliation import reconcile


def evaluate_extraction() -> tuple[pd.DataFrame, dict]:
    cases = pd.read_csv(ROOT / "data" / "extraction_cases.csv", keep_default_na=False)
    rows = []
    for case in cases.itertuples(index=False):
        output = {x["field"]: x for x in extract_fields(case.note_text)}[case.field]
        state_correct = output["answerability"] == case.expected_answerability
        value_correct = (
            str(output["value"]).upper() == str(case.expected_value).upper()
            if case.expected_answerability == ANSWERABLE else output["value"] is None
        )
        quote_verified = (
            isinstance(output["evidence_quote"], str)
            and case.note_text[output["start_offset"]:output["end_offset"]] == output["evidence_quote"]
        ) if case.expected_answerability == ANSWERABLE else True
        rows.append({
            "case_id": case.case_id, "field": case.field,
            "expected_answerability": case.expected_answerability,
            "predicted_answerability": output["answerability"],
            "state_correct": state_correct,
            "expected_value": case.expected_value,
            "predicted_value": output["value"],
            "value_correct": value_correct,
            "quote_verified": quote_verified,
            "evidence_quote": output["evidence_quote"],
        })
    df = pd.DataFrame(rows)
    expected_counts = cases["expected_answerability"].value_counts().to_dict()
    summary = {
        "examples": len(df),
        "expected_answerability_counts": {str(k): int(v) for k, v in expected_counts.items()},
        "state_correct": int(df["state_correct"].sum()),
        "value_correct": int(df["value_correct"].sum()),
        "answerable_examples": int((df["expected_answerability"] == ANSWERABLE).sum()),
        "answerable_quotes_verified": int(df.loc[df["expected_answerability"] == ANSWERABLE, "quote_verified"].sum()),
        "note": "Hand-authored parser fixtures; not a real-world model benchmark.",
    }
    return df, summary


def build_demo_outputs() -> dict:
    internal = pd.read_csv(ROOT / "data" / "internal_trades.csv", keep_default_na=False)
    external = pd.read_csv(ROOT / "data" / "external_trades.csv", keep_default_na=False)
    result = reconcile(internal, external)
    ROOT.joinpath("examples").mkdir(exist_ok=True)
    result["matches"].to_csv(ROOT / "examples" / "demo_reconciliation_matches.csv", index=False)
    result["unmatched_internal"].to_csv(ROOT / "examples" / "demo_unmatched_internal.csv", index=False)
    result["unmatched_external"].to_csv(ROOT / "examples" / "demo_unmatched_external.csv", index=False)
    result["duplicates"].to_csv(ROOT / "examples" / "demo_duplicate_groups.csv", index=False)
    result["evidence"].to_csv(ROOT / "examples" / "demo_evidence_ledger.csv", index=False)

    from exception_classifier import train_exception_classifier, classify_with_context
    model = train_exception_classifier(ROOT / "data" / "exception_training.csv")
    sentence_rows = []
    for row in external.itertuples(index=False):
        note = getattr(row, "confirmation_text", "")
        if isinstance(note, str) and note.strip():
            segments = classify_with_context(model, note, context_radius=1)
            if not segments.empty:
                segments.insert(0, "external_trade_id", str(row.trade_id))
                sentence_rows.append(segments)
    if sentence_rows:
        pd.concat(sentence_rows, ignore_index=True).to_csv(ROOT / "examples" / "demo_sentence_screening.csv", index=False)

    counts = result["matches"]["status"].value_counts().to_dict() if not result["matches"].empty else {}
    summary = {
        "internal_rows": len(internal),
        "external_rows": len(external),
        "match_status_counts": {str(k): int(v) for k, v in counts.items()},
        "unmatched_internal_rows": len(result["unmatched_internal"]),
        "unmatched_external_rows": len(result["unmatched_external"]),
        "duplicate_groups": len(result["duplicates"]),
        "evidence_answerability_counts": {str(k): int(v) for k, v in result["evidence"]["answerability"].value_counts().to_dict().items()},
        "note": "All trade rows and confirmation notes are synthetic examples.",
    }
    return summary


def main():
    (ROOT / "evaluation").mkdir(exist_ok=True)
    extraction_df, extraction_summary = evaluate_extraction()
    extraction_df.to_csv(ROOT / "evaluation" / "answerability_fixture_results.csv", index=False)
    (ROOT / "evaluation" / "answerability_fixture_summary.json").write_text(
        json.dumps(extraction_summary, indent=2), encoding="utf-8"
    )

    model = train_exception_classifier(ROOT / "data" / "exception_training.csv")
    predictions, classifier_summary = evaluate_classifier(model, ROOT / "data" / "exception_evaluation.csv")
    predictions.to_csv(ROOT / "evaluation" / "exception_classifier_predictions.csv", index=False)
    (ROOT / "evaluation" / "exception_classifier_summary.json").write_text(
        json.dumps(classifier_summary, indent=2), encoding="utf-8"
    )

    # Publish a compact reviewer-friendly register with correct and failed examples.
    selected_case_ids = [
        "EV0001", "EV0005", "EV0009", "EV0013", "EV0017", "EV0021", "EV0025", "EV0029",
        "EM0001", "EM0002", "EM0003", "EM0004", "EM0005", "EM0006", "EM0007",
    ]
    scenario_labels = {
        "EV0001": "Security identifier mismatch",
        "EV0005": "Execution date mismatch",
        "EV0009": "BUY / SELL side mismatch",
        "EV0013": "Quantity mismatch",
        "EV0017": "Price mismatch",
        "EV0021": "Settlement date mismatch",
        "EV0025": "Account allocation mismatch",
        "EV0029": "Duplicate booking",
        "EM0001": "Quantity and currency breaks in one note",
        "EM0002": "Account and settlement breaks in one note",
        "EM0003": "Duplicate booking plus side break",
        "EM0004": "Missing link plus security mismatch",
        "EM0005": "Security identity and trade-date breaks",
        "EM0006": "Quantity and account allocation breaks",
        "EM0007": "Settlement and price/currency breaks",
    }
    case_register = predictions[predictions["case_id"].astype(str).isin(selected_case_ids)].copy()
    case_register["scenario_summary"] = case_register["case_id"].map(scenario_labels)
    case_register = case_register[[
        "case_id", "group_id", "scenario_summary", "text", "expected_exception",
        "predicted_exception", "raw_svm_detector_flag", "expected_labels", "predicted_labels",
        "exact_set_match", "false_positive_labels", "false_negative_labels",
    ]]
    (ROOT / "examples").mkdir(exist_ok=True)
    case_register.to_csv(ROOT / "examples" / "prototype_test_cases.csv", index=False)

    reconciliation_summary = build_demo_outputs()
    (ROOT / "evaluation" / "trade_reconciliation_demo_summary.json").write_text(
        json.dumps(reconciliation_summary, indent=2), encoding="utf-8"
    )

    print("ANSWERABILITY FIXTURE")
    print(json.dumps(extraction_summary, indent=2))
    print("\nTWO-STAGE EXCEPTION CLASSIFIER")
    for key in [
        "training_rows", "positive_training_rows", "validation_rows_for_threshold_selection",
        "evaluation_rows",
    ]:
        print(f"{key}: {classifier_summary[key]}")
    print("raw_svm_detection_precision: " + str(classifier_summary["raw_svm_detection_precision"]))
    print("raw_svm_detection_recall: " + str(classifier_summary["raw_svm_detection_recall"]))
    print("raw_svm_detection_f1: " + str(classifier_summary["raw_svm_detection_f1"]))
    print("postprocessed_detection_precision: " + str(classifier_summary["postprocessed_detection_precision"]))
    print("postprocessed_detection_recall: " + str(classifier_summary["postprocessed_detection_recall"]))
    print("postprocessed_detection_f1: " + str(classifier_summary["postprocessed_detection_f1"]))
    for key in [
        "end_to_end_category_micro_precision", "end_to_end_category_micro_recall",
        "end_to_end_category_micro_f1", "end_to_end_category_macro_f1",
        "conditional_category_macro_f1_on_true_exceptions", "exact_label_set_accuracy",
    ]:
        print(f"{key}: {classifier_summary[key]}")
    print("\nPER-LABEL PERFORMANCE")
    print(json.dumps(classifier_summary["per_label_end_to_end"], indent=2))
    print("\nTRADE RECONCILIATION FIXTURE")
    print(json.dumps(reconciliation_summary, indent=2))


if __name__ == "__main__":
    main()
