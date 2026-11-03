"""Create reproducible, fully synthetic trade / exception-classification fixtures."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

LABEL_TEMPLATES = {
    "SECURITY_IDENTITY": [
        "Security identifier differs across systems: internal ISIN US0378331005, broker ISIN US5949181045.",
        "OMS security code ABCD123456 does not match the custodian security code WXYZ987654.",
        "Instrument reference is inconsistent; broker reports ID 1002003004 while internal system reports ID 1002003999.",
        "The two records point to different instruments because their security identifiers do not agree.",
    ],
    "TRADE_DATE": [
        "Execution date differs: internal trade date 2026-10-20, broker trade date 2026-10-21.",
        "The OMS booked the transaction on 20 October but the broker confirmation says 21 October.",
        "Trade date is inconsistent between the two sources; the external record is one day later.",
        "The execution timestamp in the internal file does not match the date on the confirmation.",
    ],
    "SIDE": [
        "Buy sell direction conflict: internal transaction is BUY but external confirmation is SELL.",
        "The OMS shows a purchase while the broker reports a sale for the same security.",
        "Trade direction is reversed across sources; BUY and SELL do not agree.",
        "One system records a buy order and the other records a sell execution.",
    ],
    "QUANTITY": [
        "Quantity mismatch: internal quantity 100 shares, broker quantity 95 shares.",
        "The OMS has 500 units but the custodian confirmation shows 450 units.",
        "Executed volume differs between files; the external feed reports fewer shares.",
        "Trade notional or unit count does not match the internally booked quantity.",
    ],
    "PRICE_CURRENCY": [
        "Price difference: internal execution price USD 225.50, broker execution price USD 226.10.",
        "The internal trade is denominated in USD while the external record is denominated in EUR.",
        "Broker price does not agree with the OMS price and the currency code is different.",
        "Unit price or settlement currency is inconsistent between internal and external records.",
    ],
    "SETTLEMENT_DATE": [
        "Settlement date mismatch: internal expected settlement 2026-10-21, custodian date 2026-10-23.",
        "The broker expects settlement on Friday while the internal record expects Thursday.",
        "Settlement instruction contains a different value date from the portfolio system.",
        "Expected settlement dates are inconsistent across the two trade records.",
    ],
    "ACCOUNT_ALLOCATION": [
        "Account allocation mismatch: internal account FUND_A, broker account FUND_B.",
        "The transaction is allocated to the wrong portfolio in the custodian file.",
        "Counterparty or fund allocation differs between the two systems.",
        "Broker account reference does not match the account recorded by the OMS.",
    ],
    "DUPLICATE_OR_MISSING_LINK": [
        "Duplicate booking detected: the same transaction appears twice in the external feed.",
        "The internal trade has no corresponding broker confirmation in the expected batch.",
        "Two external rows are competing for the same internal transaction reference.",
        "One side contains an orphan transaction and the other side has a missing counterpart.",
    ],
}

NEGATIVE_TEMPLATES = [
    "No discrepancy was identified: the security, trade date, side, quantity, price and account agree across both sources.",
    "The broker confirms the same instrument, BUY direction, 100 shares and USD 25.10 as the internal ledger; no exception remains.",
    "A previous date mismatch was corrected. The final trade date and settlement date now match the OMS.",
    "The analyst checked the account allocation and counterparty. Both records refer to the same fund and broker.",
    "There is one internal booking and one broker confirmation. No duplicate or missing link was found.",
    "The security identifier and description are consistent. The price and currency also agree.",
    "Both systems show the same ISIN and the same execution date; there is no break to investigate.",
    "Trade date and expected settlement date are aligned in the OMS and custodian confirmation.",
    "The BUY instruction is confirmed as BUY by the broker; the transaction direction agrees.",
    "Internal units and external units are both 250, so quantity is reconciled.",
    "The internal and external prices are both USD 19.75. No price or currency variance was found.",
    "The fund code and counterparty reference match exactly between the two files.",
    "There is exactly one internal trade and one corresponding external confirmation; no duplicate or orphan exists.",
    "The earlier quantity break has been corrected and the final values now agree across both sources.",
]

MULTI_LABEL_TEMPLATES = [
    ("The broker reports 90 shares rather than 100 and an execution price of USD 41.20 rather than USD 40.90.", "QUANTITY|PRICE_CURRENCY"),
    ("Trade date differs and the settlement value date is also inconsistent with the internal booking.", "TRADE_DATE|SETTLEMENT_DATE"),
    ("The external record is assigned to a different portfolio and the security identifier is not the same.", "ACCOUNT_ALLOCATION|SECURITY_IDENTITY"),
    ("Two duplicate custodian rows show a sell transaction while the internal record is marked buy.", "DUPLICATE_OR_MISSING_LINK|SIDE"),
    ("The broker used a different ISIN and the trade date is one day later than the OMS date.", "SECURITY_IDENTITY|TRADE_DATE"),
    ("The order side is BUY in the internal book but SELL externally, and the unit count is also lower.", "SIDE|QUANTITY"),
    ("The broker assigned another portfolio and reports a different execution date.", "ACCOUNT_ALLOCATION|TRADE_DATE"),
    ("The execution price differs and the expected value date is two days later at the custodian.", "PRICE_CURRENCY|SETTLEMENT_DATE"),
    ("An extra custodian booking appears and its reported notional is different from the OMS quantity.", "DUPLICATE_OR_MISSING_LINK|QUANTITY"),
    ("The instrument code differs and the external execution price is also inconsistent.", "SECURITY_IDENTITY|PRICE_CURRENCY"),
    ("The fund allocation is incorrect and the external confirmation reverses the trade direction.", "ACCOUNT_ALLOCATION|SIDE"),
    ("The broker and portfolio system disagree on both the execution date and the execution price.", "TRADE_DATE|PRICE_CURRENCY"),
    ("The clearing member confirms 210 units rather than 200 and the transaction currency is EUR instead of USD.", "QUANTITY|PRICE_CURRENCY"),
    ("This execution was allocated to Fund B and the custodian value date is later than the internal settlement date.", "ACCOUNT_ALLOCATION|SETTLEMENT_DATE"),
    ("The custodian sent two copies of the trade and one record reverses the order from BUY to SELL.", "DUPLICATE_OR_MISSING_LINK|SIDE"),
    ("The broker confirmation is absent for one instrument and the security identifier differs across the two books.", "DUPLICATE_OR_MISSING_LINK|SECURITY_IDENTITY"),
    ("The security reference is different and the external execution was booked a day after the internal trade date.", "SECURITY_IDENTITY|TRADE_DATE"),
    ("The broker allocation is to another portfolio and the external file contains 1,450 units instead of 1,500.", "ACCOUNT_ALLOCATION|QUANTITY"),
    ("The value date does not align with the OMS and the external execution price differs from the internal price.", "SETTLEMENT_DATE|PRICE_CURRENCY"),
    ("An extra custodian row has been booked to the wrong fund account.", "DUPLICATE_OR_MISSING_LINK|ACCOUNT_ALLOCATION"),
]

MULTI_LABEL_EVAL = [
    ("The external side has 75 shares instead of 80 and its currency is EUR instead of USD.", "QUANTITY|PRICE_CURRENCY"),
    ("The account is not the expected fund and the broker used a different settlement date.", "ACCOUNT_ALLOCATION|SETTLEMENT_DATE"),
    ("The OMS says BUY, but the external duplicate record says SELL.", "DUPLICATE_OR_MISSING_LINK|SIDE"),
    ("A missing counterparty confirmation also contains a different security code.", "DUPLICATE_OR_MISSING_LINK|SECURITY_IDENTITY"),
    ("The broker security code is not the same and the execution date does not line up with the book.", "SECURITY_IDENTITY|TRADE_DATE"),
    ("The portfolio has recorded 1,500 units while the external party shows 1,450, with a separate account allocated.", "QUANTITY|ACCOUNT_ALLOCATION"),
    ("The settlement value date and price currency both differ from the internal booking.", "SETTLEMENT_DATE|PRICE_CURRENCY"),
    ("The custodian sent a duplicate transaction and the instrument is allocated to another fund.", "DUPLICATE_OR_MISSING_LINK|ACCOUNT_ALLOCATION"),
]


def _variants(text: str, group: str, prefix: str, rows: list[dict], counter: list[int], labels: str):
    forms = [text, "Please review this exception. " + text, text.replace("internal", "in-house").replace("broker", "custodian")]
    for variant, form in enumerate(forms, 1):
        rows.append({"case_id": f"{prefix}{counter[0]:04d}", "group_id": group,
                     "text": form, "labels": labels})
        counter[0] += 1


def main():
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    counter = [1]
    for label, templates in LABEL_TEMPLATES.items():
        for family_idx, template in enumerate(templates, 1):
            _variants(template, f"{label}_TRAIN_{family_idx}", "TR", train_rows, counter, label)
        for family_idx, template in enumerate(templates[:2], 1):
            variants = [template.replace("internal", "portfolio").replace("broker", "clearing member"),
                        "Please investigate: " + template.replace("the two sources", "the two feeds")]
            for v in variants:
                eval_rows.append({"case_id": f"EV{len(eval_rows)+1:04d}",
                                  "group_id": f"{label}_HOLDOUT_{family_idx}", "text": v, "labels": label})

    for idx, (text, labels) in enumerate(MULTI_LABEL_TEMPLATES, 1):
        _variants(text, f"MULTI_TRAIN_{idx}", "TM", train_rows, counter, labels)
    for idx, (text, labels) in enumerate(MULTI_LABEL_EVAL, 1):
        eval_rows.append({"case_id": f"EM{idx:04d}", "group_id": f"MULTI_HOLDOUT_{idx}", "text": text, "labels": labels})

    # Negative examples include negation, corrected breaks and familiar exception
    # vocabulary in a non-exception context. They are not silently labelled as breaks.
    for family_idx, template in enumerate(NEGATIVE_TEMPLATES, 1):
        _variants(template, f"NEG_TRAIN_{family_idx}", "TN", train_rows, counter, "")
    for idx, text in enumerate([
        "All trade fields have been reconciled and no breaks remain in the batch.",
        "The prior quantity discrepancy was corrected; both feeds now show 250 units.",
        "One broker confirmation corresponds to one internal trade and the security and account agree.",
        "The trade date and settlement date are the same in the portfolio system and custodian record.",
        "Both sides show BUY, the quantity and price agree, and the counterparty allocation is correct.",
        "No security mismatch, date discrepancy, side break or duplicate was found in the final review.",
        "A previous price variance was corrected; the final currency and price are identical.",
        "The custodian confirmation is present and linked to one unique internal transaction.",
        "There is no duplicate, no orphan and one corresponding external confirmation for each internal trade.",
        "The two feeds contain the same account, same side and same quantity, so there is no allocation or quantity break.",
    ], 1):
        eval_rows.append({"case_id": f"EN{idx:04d}", "group_id": f"NEG_HOLDOUT_{idx}", "text": text, "labels": ""})

    for filename, rows in [("exception_training.csv", train_rows), ("exception_evaluation.csv", eval_rows)]:
        path = ROOT / "data" / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "group_id", "text", "labels"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Created {len(train_rows)} training and {len(eval_rows)} held-out evaluation examples; "
          f"positive train={sum(bool(x['labels']) for x in train_rows)}, negative train={sum(not x['labels'] for x in train_rows)}.")


if __name__ == "__main__":
    main()
