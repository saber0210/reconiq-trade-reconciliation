# ReconIQ synthetic exception annotation guidelines

## Unit of annotation

One row is one short exception narrative or one sentence-like segment describing a potential break between an internal trade record and an external broker / custodian record. A segment can receive more than one label.

## General labelling rules

- Apply a category only when the text states or strongly implies a discrepancy, duplicate, orphan or missing counterpart.
- Do not label a field merely because it is mentioned. “Quantity: 100” is not a quantity break without a conflicting value or exception statement.
- Do not label an explicitly reconciled / corrected item as an active exception. Examples: “No discrepancy remains”; “the previous quantity break was corrected and both feeds now agree.”
- Use multiple labels when multiple distinct issues are present in the same segment.
- Do not infer a missing value, instrument, account or trade relationship from general financial knowledge.
- If a phrase is unclear or insufficiently supported, leave it for human review; do not force a category.

## Category definitions

| Label | Apply when | Do not apply when |
|---|---|---|
| `SECURITY_IDENTITY` | ISIN, CUSIP, security code or instrument description points to different securities | Only one identifier is shown and no conflict is described |
| `TRADE_DATE` | Internal and external execution / trade dates do not agree | Dates are merely stated without a mismatch |
| `SIDE` | BUY / SELL or purchase / sale direction is reversed or inconsistent | Text only says the trade is a buy or a sell |
| `QUANTITY` | Shares, units, volume or notional differs between records | A single quantity is stated without a comparison |
| `PRICE_CURRENCY` | Price or currency differs across records | Price and currency are simply listed |
| `SETTLEMENT_DATE` | Settlement / value date differs | A settlement date is present with no comparison |
| `ACCOUNT_ALLOCATION` | Fund, account, portfolio or counterparty allocation differs | An account or counterparty is merely named |
| `DUPLICATE_OR_MISSING_LINK` | Duplicate / repeated booking, orphan, missing broker confirmation, absent counterpart or competing record is described | A unique one-to-one counterpart is explicitly confirmed |

## Multi-label examples

- “The broker reports 90 shares rather than 100 and price is USD 41.20 instead of USD 40.90.” -> `QUANTITY|PRICE_CURRENCY`.
- “The OMS is BUY; the custodian copy is a duplicate and is marked SELL.” -> `DUPLICATE_OR_MISSING_LINK|SIDE`.
- “The instrument ID differs and the broker trade date is one day later.” -> `SECURITY_IDENTITY|TRADE_DATE`.
- “All fields agree and no duplicate remains.” -> no exception labels.

## Annotation provenance and quality

The included 198 training rows and 50 held-out rows are synthetic wording templates generated for software testing. They were not independently labelled by finance professionals, and no inter-annotator agreement is claimed. For a real corpus, at least two independent operations reviewers should apply this guide, disagreements should be adjudicated, and category-wise agreement should be reported before model training.
