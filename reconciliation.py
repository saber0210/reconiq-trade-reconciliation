"""Evidence-first two-way trade reconciliation with explainable rule mapping."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidence_extraction import ANSWERABLE, AMBIGUOUS, PATTERNS, extract_fields, canonical_value

REQUIRED = {"trade_id", "trade_date", "security_id", "security_description", "side", "quantity", "price", "currency"}
OPTIONAL = {"settlement_date", "account", "counterparty", "confirmation_text"}
DIMENSION_WEIGHTS = {
    "R1_IDENTITY": 0.25,
    "R2_TRADE_DATE": 0.15,
    "R3_SIDE": 0.10,
    "R4_QUANTITY": 0.15,
    "R5_PRICE_CURRENCY": 0.15,
    "R6_SETTLEMENT_ALLOCATION": 0.20,
}
REQUIRED_DIMS = ["R1_IDENTITY", "R2_TRADE_DATE", "R3_SIDE", "R4_QUANTITY", "R5_PRICE_CURRENCY"]
EXCEPTION_LABELS = {
    "R1_IDENTITY": "SECURITY_IDENTITY",
    "R2_TRADE_DATE": "TRADE_DATE",
    "R3_SIDE": "SIDE",
    "R4_QUANTITY": "QUANTITY",
    "R5_PRICE_CURRENCY": "PRICE_CURRENCY",
}


def missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def clean_text(v: Any) -> str:
    if missing(v):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(v).upper())).strip()


def norm(field: str, v: Any) -> Any:
    if missing(v):
        return None
    if field in {"trade_date", "settlement_date"}:
        value = str(v).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except ValueError:
                return None
        vals = set()
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                vals.add(datetime.strptime(value, fmt).date().isoformat())
            except ValueError:
                continue
        return next(iter(vals)) if len(vals) == 1 else None
    if field in {"quantity", "price"}:
        x = pd.to_numeric(v, errors="coerce")
        return None if pd.isna(x) else float(x)
    if field in {"security_id", "side", "currency"}:
        return str(v).strip().upper()
    if field in {"account", "counterparty", "security_description"}:
        return clean_text(v)
    return str(v).strip()


def compare(a: Any, b: Any, field: str, tolerance: float = 0.01) -> str:
    x, y = norm(field, a), norm(field, b)
    if x is None or y is None:
        return "UNKNOWN"
    if field == "price":
        return "AGREES" if abs(float(x) - float(y)) <= tolerance else "CONFLICTS"
    return "AGREES" if x == y else "CONFLICTS"


def profile(df: pd.DataFrame, source: str) -> dict[str, Any]:
    missing_columns = sorted(REQUIRED - set(df.columns))
    missing_values = {}
    for c in sorted(REQUIRED & set(df.columns)):
        missing_values[c] = int(df[c].isna().sum() + df[c].astype(str).str.strip().eq("").sum())
    id_duplicates = int(df["trade_id"].astype(str).duplicated(keep=False).sum()) if "trade_id" in df.columns else 0
    return {"source": source, "rows": len(df), "missing_required_columns": ", ".join(missing_columns) or "None",
            "missing_required_values": missing_values, "duplicate_source_trade_id_rows": id_duplicates}


def _prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
    absent = REQUIRED - set(df.columns)
    if absent:
        raise ValueError(f"{source} missing required columns: {', '.join(sorted(absent))}")
    out = df.copy().reset_index(drop=True)
    for c in OPTIONAL:
        if c not in out.columns:
            out[c] = ""
    out["_source"] = source
    out["_row_number"] = np.arange(2, len(out) + 2)
    for c in ["security_id", "security_description", "side", "currency", "account", "counterparty"]:
        out[f"{c}_norm"] = out[c].map(lambda v, field=c: norm(field, v))
    for c in ["trade_date", "settlement_date", "quantity", "price"]:
        out[f"{c}_norm"] = out[c].map(lambda v, field=c: norm(field, v))
    return out


def _extract_external(external_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = external_df.copy().reset_index(drop=True)
    for c in OPTIONAL:
        if c not in enriched.columns:
            enriched[c] = ""
    evidence_rows = []
    for i, row in enriched.iterrows():
        note = str(row.get("confirmation_text", "") or "")
        extracted = extract_fields(note)
        conflicts, ambiguous = [], []
        for item in extracted:
            item = dict(item)
            item["external_trade_id"] = str(row.get("trade_id", ""))
            evidence_rows.append(item)
            field = item["field"]
            if item["answerability"] == AMBIGUOUS:
                ambiguous.append(field)
                continue
            if item["answerability"] != ANSWERABLE:
                continue
            current = enriched.at[i, field] if field in enriched.columns else ""
            if missing(current):
                enriched.at[i, field] = item["value"]
            else:
                old, new = norm(field, current), norm(field, item["value"])
                differs = (abs(float(old) - float(new)) > 0.01) if field == "price" and old is not None and new is not None else old != new
                if differs:
                    conflicts.append(field)
        enriched.at[i, "_note_conflicts"] = "|".join(sorted(set(conflicts)))
        enriched.at[i, "_note_ambiguous"] = "|".join(sorted(set(ambiguous)))
    return enriched, pd.DataFrame(evidence_rows)


def _dimension_vector(i: pd.Series, e: pd.Series, sim: float, id_exact: bool) -> dict[str, str]:
    if id_exact:
        identity = "AGREES"
    elif i["security_id_norm"] is not None and e["security_id_norm"] is not None:
        identity = "CONFLICTS"
    else:
        identity = "UNKNOWN"
    trade_date = compare(i["trade_date_norm"], e["trade_date_norm"], "trade_date")
    side = compare(i["side_norm"], e["side_norm"], "side")
    quantity = compare(i["quantity_norm"], e["quantity_norm"], "quantity")
    price = compare(i["price_norm"], e["price_norm"], "price")
    currency = compare(i["currency_norm"], e["currency_norm"], "currency")
    price_currency = "CONFLICTS" if "CONFLICTS" in {price, currency} else ("AGREES" if price == currency == "AGREES" else "UNKNOWN")
    extra = [
        compare(i["settlement_date_norm"], e["settlement_date_norm"], "settlement_date"),
        compare(i["account_norm"], e["account_norm"], "account"),
        compare(i["counterparty_norm"], e["counterparty_norm"], "counterparty"),
    ]
    if "CONFLICTS" in extra:
        r6 = "CONFLICTS"
    elif all(x == "AGREES" for x in extra):
        r6 = "AGREES"
    else:
        r6 = "UNKNOWN"
    return {
        "R1_IDENTITY": identity, "R2_TRADE_DATE": trade_date, "R3_SIDE": side,
        "R4_QUANTITY": quantity, "R5_PRICE_CURRENCY": price_currency,
        "R6_SETTLEMENT_ALLOCATION": r6,
        "SETTLEMENT_DATE_DETAIL": extra[0], "ACCOUNT_DETAIL": extra[1], "COUNTERPARTY_DETAIL": extra[2],
    }


def _rank(vector: dict[str, str], similarity: float, id_exact: bool) -> float:
    known = sum(DIMENSION_WEIGHTS[k] for k in DIMENSION_WEIGHTS if vector[k] != "UNKNOWN")
    agrees = sum(DIMENSION_WEIGHTS[k] for k in DIMENSION_WEIGHTS if vector[k] == "AGREES")
    evidence_agreement = agrees / known if known else 0.0
    identity_signal = 1.0 if id_exact else max(0.0, min(1.0, similarity))
    return round(0.85 * evidence_agreement + 0.15 * identity_signal, 4)


def _duplicate_indices(df: pd.DataFrame) -> set[int]:
    keys = ["trade_date_norm", "security_id_norm", "security_description_norm", "side_norm", "quantity_norm", "price_norm", "currency_norm", "account_norm"]
    if any(k not in df.columns for k in keys):
        return set()
    return set(df.index[df.duplicated(subset=keys, keep=False)].tolist())


def _duplicate_groups(df: pd.DataFrame, source: str) -> pd.DataFrame:
    keys = ["trade_date_norm", "security_id_norm", "security_description_norm", "side_norm", "quantity_norm", "price_norm", "currency_norm", "account_norm"]
    mask = df.duplicated(subset=keys, keep=False)
    out = []
    for _, group in df.loc[mask].groupby(keys, dropna=False):
        out.append({"source": source, "trade_ids": ", ".join(group["trade_id"].astype(str)), "duplicate_count": len(group)})
    return pd.DataFrame(out, columns=["source", "trade_ids", "duplicate_count"])


def reconcile(internal_df: pd.DataFrame, external_df: pd.DataFrame,
              description_candidate_threshold: float = 0.18) -> dict[str, pd.DataFrame]:
    prof_i, prof_e = profile(internal_df, "internal"), profile(external_df, "external")
    if prof_i["missing_required_columns"] != "None":
        raise ValueError(f"internal missing required columns: {prof_i['missing_required_columns']}")
    if prof_e["missing_required_columns"] != "None":
        raise ValueError(f"external missing required columns: {prof_e['missing_required_columns']}")
    enriched_e, evidence = _extract_external(external_df)
    internal, external = _prepare(internal_df, "internal"), _prepare(enriched_e, "external")
    text_all = pd.concat([internal["security_description_norm"], external["security_description_norm"]], ignore_index=True).fillna("")
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
        matrix = vec.fit_transform(text_all)
        sims = cosine_similarity(matrix[:len(internal)], matrix[len(internal):])
    except ValueError:
        sims = np.zeros((len(internal), len(external)))

    pairs = []
    for i, ir in internal.iterrows():
        for j, er in external.iterrows():
            id_exact = bool(ir["security_id_norm"] and er["security_id_norm"] and ir["security_id_norm"] == er["security_id_norm"])
            sim = float(sims[i, j])
            if not id_exact and sim < description_candidate_threshold:
                continue
            dims = _dimension_vector(ir, er, sim, id_exact)
            rank_score = _rank(dims, sim, id_exact)
            pairs.append({"internal_index": i, "external_index": j,
                          "internal_trade_id": ir["trade_id"], "external_trade_id": er["trade_id"],
                          "id_exact": id_exact, "description_similarity": round(sim, 4),
                          "rank_score": rank_score, **dims})
    candidates = pd.DataFrame(pairs)
    duplicates = pd.concat([_duplicate_groups(internal, "internal"), _duplicate_groups(external, "external")], ignore_index=True)
    empty_matches = pd.DataFrame(columns=["internal_trade_id", "external_trade_id", "status", "rank_score", "reason", "rule_exception_labels"])
    if candidates.empty:
        return {"matches": empty_matches, "candidates": candidates,
                "unmatched_internal": internal_df.copy(), "unmatched_external": external_df.copy(),
                "duplicates": duplicates, "evidence": evidence,
                "profile": pd.DataFrame([prof_i, prof_e])}

    ordered = candidates.sort_values(["rank_score", "description_similarity"], ascending=False)
    best_i = ordered.drop_duplicates("internal_index", keep="first").set_index("internal_index")
    best_e = ordered.drop_duplicates("external_index", keep="first").set_index("external_index")
    dup_i, dup_e = _duplicate_indices(internal), _duplicate_indices(external)
    rows, chosen_e = [], set()
    for i, best in best_i.iterrows():
        j = int(best["external_index"])
        chosen_e.add(j)
        ir, er = internal.iloc[int(i)], external.iloc[j]
        rank = float(best["rank_score"])
        i_candidates = ordered[ordered.internal_index == int(i)]
        e_candidates = ordered[ordered.external_index == j]
        ambiguous = (len(i_candidates[i_candidates.rank_score >= rank - 0.035]) > 1 or
                     len(e_candidates[e_candidates.rank_score >= rank - 0.035]) > 1 or
                     int(best_e.loc[j, "internal_index"]) != int(i))
        duplicate = int(i) in dup_i or j in dup_e
        vector = {k: str(best[k]) for k in DIMENSION_WEIGHTS}
        detail_states = {k: str(best[k]) for k in ["SETTLEMENT_DATE_DETAIL", "ACCOUNT_DETAIL", "COUNTERPARTY_DETAIL"]}
        conflicts = [k for k, v in vector.items() if v == "CONFLICTS"]
        unknown_required = [k for k in REQUIRED_DIMS if vector[k] == "UNKNOWN"]
        unknown_support = vector["R6_SETTLEMENT_ALLOCATION"] == "UNKNOWN"
        note_conflicts = [x for x in str(er.get("_note_conflicts", "")).split("|") if x]
        note_ambiguous = [x for x in str(er.get("_note_ambiguous", "")).split("|") if x]
        reasons = []
        if not bool(best["id_exact"]): reasons.append("Security ID not exact; description used only for candidate discovery")
        for k, v in vector.items():
            if v == "CONFLICTS": reasons.append(f"{k}: conflict")
            elif v == "UNKNOWN": reasons.append(f"{k}: evidence unknown")
        if note_conflicts: reasons.append("Structured confirmation conflicts with note: " + ", ".join(note_conflicts))
        if note_ambiguous: reasons.append("Ambiguous note field(s): " + ", ".join(note_ambiguous))
        if ambiguous: reasons.append("Competing candidate / non-mutual-best link")
        if duplicate: reasons.append("Duplicate business key detected")

        if duplicate or ambiguous:
            status = "REVIEW - AMBIGUOUS / DUPLICATE"
        elif note_conflicts or note_ambiguous:
            status = "REVIEW - NARRATIVE EVIDENCE"
        elif conflicts:
            status = "REVIEW - FIELD CONFLICT"
        elif unknown_required or unknown_support:
            status = "REVIEW - UNKNOWN EVIDENCE"
        elif bool(best["id_exact"]) and all(vector[k] == "AGREES" for k in DIMENSION_WEIGHTS):
            status = "MATCHED - EXACT"
        else:
            status = "REVIEW - SUGGESTED"

        issue_labels = []
        for dim, label in EXCEPTION_LABELS.items():
            if vector[dim] == "CONFLICTS": issue_labels.append(label)
        if detail_states["SETTLEMENT_DATE_DETAIL"] == "CONFLICTS":
            issue_labels.append("SETTLEMENT_DATE")
        if detail_states["ACCOUNT_DETAIL"] == "CONFLICTS" or detail_states["COUNTERPARTY_DETAIL"] == "CONFLICTS":
            issue_labels.append("ACCOUNT_ALLOCATION")
        if duplicate or ambiguous: issue_labels.append("DUPLICATE_OR_MISSING_LINK")
        if note_ambiguous or unknown_required or unknown_support: issue_labels.append("MISSING_OR_AMBIGUOUS_EVIDENCE")
        if note_conflicts: issue_labels.append("NARRATIVE_STRUCTURED_CONFLICT")

        note_text = str(er.get("confirmation_text", "") or "")
        note_rows = evidence[evidence.external_trade_id.astype(str) == str(er["trade_id"])] if not evidence.empty else pd.DataFrame()
        quotes = [f"{r.field}: {r.evidence_quote}" for r in note_rows.itertuples()
                  if getattr(r, "answerability", "") == ANSWERABLE and isinstance(getattr(r, "evidence_quote", None), str)]
        rows.append({
            "internal_trade_id": str(ir["trade_id"]), "external_trade_id": str(er["trade_id"]),
            "status": status, "rank_score": rank,
            **vector,
            "reason": "; ".join(reasons) if reasons else "All R1-R6 evidence dimensions agree.",
            "rule_exception_labels": "|".join(sorted(set(issue_labels))),
            "confirmation_text": note_text,
            "evidence_quotes": " | ".join(quotes),
            "internal_security": ir["security_description"], "external_security": er["security_description"],
            "internal_quantity": ir["quantity"], "external_quantity": er["quantity"],
            "internal_price": ir["price"], "external_price": er["price"],
            "human_decision": "Pending",
        })

    matches = pd.DataFrame(rows).sort_values(["status", "internal_trade_id"]).reset_index(drop=True)
    cand_i = set(ordered.internal_index.astype(int))
    cand_e = set(ordered.external_index.astype(int))
    unmatched_i = internal.loc[~internal.index.isin(cand_i), internal_df.columns.intersection(internal.columns)].copy()
    unmatched_e = external.loc[~external.index.isin(cand_e), external_df.columns.intersection(external.columns)].copy()
    return {"matches": matches,
            "candidates": ordered.drop(columns=["internal_index", "external_index"]).reset_index(drop=True),
            "unmatched_internal": unmatched_i, "unmatched_external": unmatched_e,
            "duplicates": duplicates, "evidence": evidence,
            "profile": pd.DataFrame([prof_i, prof_e])}
