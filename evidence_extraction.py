
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
import re

ANSWERABLE = "ANSWERABLE"
NOT_FOUND = "NOT_FOUND"
AMBIGUOUS = "AMBIGUOUS"
INVALID_EVIDENCE = "INVALID_EVIDENCE"

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "trade_date": [re.compile(r"\btrade\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})", re.I)],
    "settlement_date": [re.compile(r"\bsettlement\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})", re.I)],
    "side": [re.compile(r"\b(?:side|action)\s*[:=]\s*(BUY|SELL)\b", re.I)],
    "quantity": [re.compile(r"\b(?:quantity|qty|shares|units)\s*[:=]\s*([0-9][0-9,]*(?:\.\d+)?)", re.I)],
    "price": [re.compile(r"\b(?:execution\s+price|trade\s+price|price)\s*[:=]\s*(?:(?:USD|EUR|GBP|INR|JPY|CHF|CAD|AUD)\s*)?([0-9][0-9,]*(?:\.\d+)?)", re.I)],
    "currency": [re.compile(r"\bcurrency\s*[:=]\s*([A-Z]{3})\b", re.I),
                 re.compile(r"\bprice\s*[:=]\s*(USD|EUR|GBP|INR|JPY|CHF|CAD|AUD)\s+[0-9]", re.I)],
    "security_id": [re.compile(r"\b(?:security\s+id|instrument\s+id|isin|cusip)\s*[:=]\s*([A-Z0-9]{8,16})\b", re.I)],
    "security_description": [re.compile(r"\b(?:security|instrument|description)\s*[:=]\s*([^;\n|]+)", re.I)],
    "account": [re.compile(r"\b(?:account|fund|portfolio)\s*[:=]\s*([A-Z0-9_-]+)\b", re.I)],
    "counterparty": [re.compile(r"\b(?:counterparty|broker|custodian)\s*[:=]\s*([A-Z0-9_-]+)\b", re.I)],
}


def _date_values(raw: str) -> list[str]:
    value = raw.strip().strip(" ,.")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return [datetime.strptime(value, "%Y-%m-%d").date().isoformat()]
        except ValueError:
            return []
    found = set()
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            found.add(datetime.strptime(value, fmt).date().isoformat())
        except ValueError:
            pass
    return sorted(found)


def canonical_value(field: str, raw: Any) -> Any:
    value = str(raw).strip().strip(" ,.")
    if field in {"trade_date", "settlement_date"}:
        vals = _date_values(value)
        if len(vals) != 1:
            raise ValueError("Date is invalid or ambiguous.")
        return vals[0]
    if field == "side":
        return value.upper()
    if field == "quantity":
        n = float(value.replace(",", ""))
        return int(n) if n.is_integer() else n
    if field == "price":
        return round(float(value.replace(",", "")), 8)
    if field in {"security_id", "currency", "account", "counterparty"}:
        return value.upper()
    if field == "security_description":
        return re.sub(r"\s+", " ", value).strip()
    return value


@dataclass
class Evidence:
    field: str
    answerability: str
    value: Any = None
    evidence_quote: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    method: str = "local_labelled_field_parser"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_one(text: str, field: str) -> Evidence:
    matches = []
    for pattern in PATTERNS[field]:
        for m in pattern.finditer(text):
            raw = m.group(1)
            try:
                if field in {"trade_date", "settlement_date"}:
                    values = _date_values(raw)
                    if len(values) == 0:
                        continue
                    if len(values) > 1:
                        for value in values:
                            matches.append((value, m.group(0), m.start(), m.end()))
                        continue
                    value = values[0]
                else:
                    value = canonical_value(field, raw)
            except (ValueError, TypeError):
                continue
            item = (value, m.group(0), m.start(), m.end())
            if item not in matches:
                matches.append(item)
    if not matches:
        return Evidence(field, NOT_FOUND, note="No explicit field value found in the supplied note.")
    distinct = {str(x[0]) for x in matches}
    if len(distinct) > 1:
        quote = " | ".join(dict.fromkeys(x[1] for x in matches))
        return Evidence(field, AMBIGUOUS, evidence_quote=quote,
                        start_offset=matches[0][2], end_offset=matches[-1][3],
                        note="Multiple valid values or interpretations were found; no value was selected.")
    value, quote, start, end = matches[0]
    return Evidence(field, ANSWERABLE, value, quote, start, end,
                    note="Value is supported by an exact source-text span.")


def extract_fields(note_text: str | None) -> list[dict[str, Any]]:
    text = "" if note_text is None else str(note_text)
    if not text.strip():
        return [Evidence(field, NOT_FOUND, note="No confirmation note supplied.").as_dict() for field in PATTERNS]
    return [_extract_one(text, field).as_dict() for field in PATTERNS]


def validate_model_payload(payload: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
    """Optional LLM guard: answerable values require a verbatim supporting quote."""
    fields = payload.get("fields", payload)
    output = []
    for field in PATTERNS:
        item = fields.get(field, {}) if isinstance(fields, dict) else {}
        if not isinstance(item, dict):
            item = {}
        state = str(item.get("answerability", NOT_FOUND)).upper()
        value = item.get("value")
        quote = item.get("evidence_quote")
        if state in {"YES", "TRUE", ANSWERABLE}:
            state = ANSWERABLE
        elif state in {"NO", "FALSE", "NOT_ANSWERABLE", NOT_FOUND}:
            state = NOT_FOUND
        elif state != AMBIGUOUS:
            state = INVALID_EVIDENCE
        if state == ANSWERABLE:
            if not isinstance(quote, str) or not quote or quote not in source_text or value is None:
                res = Evidence(field, INVALID_EVIDENCE, note="Rejected: missing or non-verbatim evidence quote.", method="model_guard")
            else:
                start = source_text.find(quote)
                extracted = _extract_one(quote, field)
                try:
                    proposed = canonical_value(field, value)
                except (ValueError, TypeError):
                    proposed = None
                if extracted.answerability != ANSWERABLE or str(extracted.value) != str(proposed):
                    res = Evidence(field, INVALID_EVIDENCE, evidence_quote=quote,
                                   start_offset=start, end_offset=start + len(quote),
                                   method="model_guard", note="Rejected: quote does not support the proposed value.")
                else:
                    res = Evidence(field, ANSWERABLE, proposed, quote, start, start + len(quote),
                                   method="model_guard", note="Accepted after exact quote and value verification.")
        elif state == AMBIGUOUS:
            res = Evidence(field, AMBIGUOUS, evidence_quote=quote if isinstance(quote, str) else None,
                           method="model_guard", note="Model marked the field ambiguous; human review required.")
        else:
            if value is not None or quote not in (None, ""):
                res = Evidence(field, INVALID_EVIDENCE, method="model_guard",
                               note="Rejected: NOT_FOUND response also supplied a value or quote.")
            else:
                res = Evidence(field, NOT_FOUND, method="model_guard", note="No field evidence returned.")
        output.append(res.as_dict())
    return output
