from pathlib import Path
import pandas as pd
import pytest

from evidence_extraction import (
    ANSWERABLE, NOT_FOUND, AMBIGUOUS, INVALID_EVIDENCE,
    extract_fields, validate_model_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = pd.read_csv(ROOT / "data" / "extraction_cases.csv", keep_default_na=False)


@pytest.mark.parametrize("case_id", CASES["case_id"].tolist())
def test_synthetic_extraction_case(case_id):
    row = CASES.loc[CASES.case_id == case_id].iloc[0]
    result = {x["field"]: x for x in extract_fields(row.note_text)}[row.field]
    assert result["answerability"] == row.expected_answerability
    if row.expected_answerability == ANSWERABLE:
        assert str(result["value"]).upper() == str(row.expected_value).upper()
        assert result["evidence_quote"] in row.note_text
        assert row.note_text[result["start_offset"]:result["end_offset"]] == result["evidence_quote"]
    else:
        assert result["value"] is None


def test_empty_note_returns_not_found_for_every_field():
    results = extract_fields("")
    assert results
    assert all(x["answerability"] == NOT_FOUND for x in results)


def test_same_repeated_value_is_still_answerable():
    result = {x["field"]: x for x in extract_fields("Quantity: 100; Quantity: 100.")} ["quantity"]
    assert result["answerability"] == ANSWERABLE
    assert result["value"] == 100


def test_invalid_calendar_date_is_not_extracted():
    result = {x["field"]: x for x in extract_fields("Trade date: 2026-02-31.")} ["trade_date"]
    assert result["answerability"] == NOT_FOUND


def test_model_answer_requires_a_verbatim_supporting_quote():
    source = "Side: BUY; Quantity: 100."
    payload = {"fields": {"quantity": {"answerability": "ANSWERABLE", "value": "100", "evidence_quote": "Quantity: 100"}}}
    result = {x["field"]: x for x in validate_model_payload(payload, source)}["quantity"]
    assert result["answerability"] == ANSWERABLE
    assert source[result["start_offset"]:result["end_offset"]] == result["evidence_quote"]


def test_model_hallucinated_quote_is_invalid_evidence():
    source = "Quantity: 100."
    payload = {"fields": {"quantity": {"answerability": "ANSWERABLE", "value": "125", "evidence_quote": "Quantity: 125"}}}
    result = {x["field"]: x for x in validate_model_payload(payload, source)}["quantity"]
    assert result["answerability"] == INVALID_EVIDENCE
    assert result["value"] is None


def test_model_not_found_with_a_value_is_invalid_evidence():
    payload = {"fields": {"settlement_date": {"answerability": "NOT_FOUND", "value": "2026-10-21", "evidence_quote": None}}}
    result = {x["field"]: x for x in validate_model_payload(payload, "No settlement date stated.")} ["settlement_date"]
    assert result["answerability"] == INVALID_EVIDENCE


def test_model_ambiguous_state_never_returns_a_selected_value():
    payload = {"fields": {"trade_date": {"answerability": "AMBIGUOUS", "value": None, "evidence_quote": "03/04/2026"}}}
    result = {x["field"]: x for x in validate_model_payload(payload, "Trade date: 03/04/2026.")} ["trade_date"]
    assert result["answerability"] == AMBIGUOUS
    assert result["value"] is None
