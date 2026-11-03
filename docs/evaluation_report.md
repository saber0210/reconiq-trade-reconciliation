# ReconIQ evaluation report - synthetic prototype only

## 1. Answerability-aware extraction fixture

| Expected state | Cases |
|---|---:|
| `ANSWERABLE` | 11 |
| `NOT_FOUND` | 6 |
| `AMBIGUOUS` | 6 |
| **Total** | **23** |

Current check: 23 / 23 statuses and extracted values match the hand-authored expectations; 11 / 11 answerable values have exact source quotes and verified offsets. This checks the local labelled-field parser, not a trained QA model.

## 2. Exception classifier fixture

### Data and split

- 198 synthetic training-pool rows: 156 exception-positive and 42 negative / clean examples.
- 30 grouped wording-family examples are reserved from fitting to select detection and per-category thresholds.
- 50 phrase-family-held-out evaluation rows: 42 exception narratives and 8 negative / clean narratives.
- Eight output categories; examples can have multiple labels.

The category classifiers are fitted only on positive exception examples, separately from the binary detector. A multi-label exact-set match requires every suggested label to match the expected set; a correct binary flag with one missing category is still an exact-set error.

### Detection performance

| Detector stage | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw Linear SVM detector | 0.9756 | 1.0000 | 0.9877 |
| After narrow clean-link post-processing | 1.0000 | 1.0000 | 1.0000 |

The post-processing rule only suppresses the duplicate / missing-link category when the note explicitly states that a unique clean counterpart exists. The perfect post-processed result is on the small synthetic fixture and should be interpreted with particular caution.

### End-to-end multi-label performance

| Metric | Result |
|---|---:|
| Micro precision | 0.9556 |
| Micro recall | 0.8958 |
| Micro F1 | 0.9247 |
| Macro precision | 0.9613 |
| Macro recall | 0.9018 |
| **Macro F1** | **0.9218** |
| Exact label-set accuracy | 0.9000 |
| Hamming loss | 0.0175 |

### Per-category results

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `SECURITY_IDENTITY` | 1.0000 | 0.6667 | 0.8000 | 6 |
| `TRADE_DATE` | 0.8333 | 1.0000 | 0.9091 | 5 |
| `SIDE` | 1.0000 | 1.0000 | 1.0000 | 5 |
| `QUANTITY` | 1.0000 | 0.8333 | 0.9091 | 6 |
| `PRICE_CURRENCY` | 0.8571 | 1.0000 | 0.9231 | 6 |
| `SETTLEMENT_DATE` | 1.0000 | 1.0000 | 1.0000 | 6 |
| `ACCOUNT_ALLOCATION` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `DUPLICATE_OR_MISSING_LINK` | 1.0000 | 0.7143 | 0.8333 | 7 |

### Error cases requiring review

Five evaluation rows did not have an exact expected-vs-predicted label-set match: `EM0003`, `EM0004`, `EM0005`, `EM0006` and `EM0007`. Open `evaluation/exception_classifier_predictions.csv` and inspect the `false_positive_labels` and `false_negative_labels` columns to see which categories were missed or over-suggested.

The fact that exact label-set accuracy is 0.90, despite binary detection F1 being near 1.0, is important: identifying that a sentence contains some exception is easier than correctly recovering every category in a multi-break narrative. This supports the paper-inspired practice of reporting detection and categorisation separately.

## 3. Deterministic reconciliation fixture

| Result | Count |
|---|---:|
| Internal trade rows | 13 |
| External trade rows | 13 |
| `MATCHED - EXACT` | 3 |
| Review candidates | 9 |
| Unmatched internal | 1 |
| Unmatched external | 1 |
| Duplicate business-key groups | 1 |

Review candidates comprise four field conflicts, three ambiguous / duplicate cases, one narrative-evidence conflict and one unknown-evidence case. The ML text classifier does not change these deterministic reconciliation statuses.

## 4. CLAUDETTE results are not a baseline to beat

CLAUDETTE's C2 (category-specific SVM combination) reported F1 0.781 and C8 (ensemble) 0.805 on a leave-one-document-out legal ToS corpus; its additional ten-contract test yielded ensemble F1 0.736. ReconIQ's 0.9218 multi-label macro F1 is computed on a small synthetic finance phrase fixture. It is **not comparable** with CLAUDETTE because the domain, annotation process, data distribution, split, class balance and metric aggregation differ. The appropriate use of CLAUDETTE here is architectural inspiration and evaluation discipline, not a score target.
