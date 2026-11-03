# Data dictionary and synthetic-data notice

Every row bundled here is synthetic. No real trades, funds, securities, counterparties, brokers, custodians, customers or accounts are represented.

## Trade inputs

`internal_trades.csv` and `external_trades.csv` contain:

- `trade_id`: source-specific row ID; not expected to match between systems.
- `trade_date`: trade execution date.
- `settlement_date`: expected settlement date.
- `security_id`: security identifier; selected external examples intentionally omit it.
- `security_description`: source-specific instrument description.
- `side`: BUY or SELL.
- `quantity`, `price`, `currency`: economic attributes.
- `account`, `counterparty`: allocation / broker / custodian fields.
- `confirmation_text` (external file): synthetic unstructured confirmation note.

## Exception classifier fixtures

`exception_training.csv` has 198 synthetic examples, 156 with one or more exception labels and 42 clean / negative examples. It includes `group_id` values used to reserve wording families for threshold validation.

`exception_evaluation.csv` has 50 phrase-family-held-out synthetic examples, including both single-label and multi-label examples plus clean / negative examples. It is a designed software fixture, not an independently collected finance corpus.

## Answerability fixtures

`extraction_cases.csv` has 23 hand-authored examples: 11 answerable, 6 not found and 6 ambiguous. It checks the local labelled-field parser and exact quote offsets; it is not a model benchmark.

## Label semantics

The eight exception categories are defined in `docs/annotation_guidelines.md`. One text row may contain multiple labels separated by `|`. The labels identify operational break types and are not severity scores or accounting conclusions.


## Owner-created challenge examples

`owner_challenge_set_template.csv` is an empty template for new examples written and labelled by the project owner. Keep these examples separate from the training and evaluation files. Do not add them to the reported metrics unless you define a new split, keep the labels fixed before model predictions are inspected, and document the evaluation method.
