from pathlib import Path
import pandas as pd
import pytest

from reconciliation import reconcile

ROOT = Path(__file__).resolve().parents[1]


def load_demo():
    internal = pd.read_csv(ROOT / "data" / "internal_trades.csv", keep_default_na=False)
    external = pd.read_csv(ROOT / "data" / "external_trades.csv", keep_default_na=False)
    return internal, external


def row_for(result, internal_id):
    rows = result["matches"].loc[result["matches"]["internal_trade_id"] == internal_id]
    assert len(rows) == 1
    return rows.iloc[0]


def test_exact_unique_trade_is_matched():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I001")["status"] == "MATCHED - EXACT"


def test_quote_verified_note_can_fill_blank_security_id():
    internal, external = load_demo()
    external.loc[external.trade_id == "E902", "security_id"] = ""
    result = reconcile(internal, external)
    assert row_for(result, "I002")["status"] == "MATCHED - EXACT"
    evidence = result["evidence"].query("external_trade_id == 'E902' and field == 'security_id'").iloc[0]
    assert evidence["answerability"] == "ANSWERABLE"
    assert evidence["evidence_quote"] == "Security ID: US5949181045"


def test_quantity_mismatch_stays_in_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    r = row_for(result, "I003")
    assert r["status"] == "REVIEW - NARRATIVE EVIDENCE"
    assert r["R4_QUANTITY"] == "CONFLICTS"


def test_price_mismatch_stays_in_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I004")["R5_PRICE_CURRENCY"] == "CONFLICTS"


def test_trade_date_conflict_stays_in_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I005")["R2_TRADE_DATE"] == "CONFLICTS"


def test_side_conflict_stays_in_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I009")["R3_SIDE"] == "CONFLICTS"


def test_currency_conflict_stays_in_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I010")["R5_PRICE_CURRENCY"] == "CONFLICTS"


def test_duplicate_external_record_is_flagged():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert not result["duplicates"].empty
    assert row_for(result, "I007")["status"] == "REVIEW - AMBIGUOUS / DUPLICATE"


def test_competing_candidates_go_to_review():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert row_for(result, "I008")["status"] == "REVIEW - AMBIGUOUS / DUPLICATE"


def test_missing_internal_and_external_links_are_reported():
    internal, external = load_demo()
    result = reconcile(internal, external)
    assert "I006" in set(result["unmatched_internal"]["trade_id"])
    assert "E999" in set(result["unmatched_external"]["trade_id"])


def test_missing_supporting_fields_are_unknown_not_agreement():
    internal, external = load_demo()
    internal = internal.drop(columns=["account", "settlement_date", "counterparty"])
    external = external.drop(columns=["account", "settlement_date", "counterparty"])
    result = reconcile(internal, external)
    assert row_for(result, "I001")["R6_SETTLEMENT_ALLOCATION"] == "UNKNOWN"
    assert row_for(result, "I001")["status"] == "REVIEW - UNKNOWN EVIDENCE"


def test_structured_ambiguous_date_is_unknown_not_locale_guessed():
    internal, external = load_demo()
    internal.loc[internal.trade_id == "I001", "trade_date"] = "03/04/2026"
    external.loc[external.trade_id == "E901", "trade_date"] = "03/04/2026"
    external.loc[external.trade_id == "E901", "confirmation_text"] = external.loc[
        external.trade_id == "E901", "confirmation_text"
    ].str.replace("Trade date: 2026-10-20; ", "", regex=False)
    result = reconcile(internal, external)
    assert row_for(result, "I001")["R2_TRADE_DATE"] == "UNKNOWN"


def test_missing_required_column_is_rejected():
    internal, external = load_demo()
    with pytest.raises(ValueError, match="quantity"):
        reconcile(internal.drop(columns=["quantity"]), external)


def test_profile_reports_duplicate_source_ids():
    internal, external = load_demo()
    internal.loc[1, "trade_id"] = internal.loc[0, "trade_id"]
    result = reconcile(internal, external)
    profile = result["profile"].query("source == 'internal'").iloc[0]
    assert profile["duplicate_source_trade_id_rows"] == 2


def test_settlement_conflict_maps_to_settlement_label_not_account():
    internal, external = load_demo()
    result = reconcile(internal, external)
    # I005/E905 has both trade-date and settlement-date conflicts in the synthetic case.
    labels = set(row_for(result, "I005")["rule_exception_labels"].split("|"))
    assert "SETTLEMENT_DATE" in labels
