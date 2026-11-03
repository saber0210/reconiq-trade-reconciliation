# ReconIQ: a prototype for trade reconciliation

**Status:** educational prototype using synthetic trade records and synthetic confirmation notes. It does not connect to a live broker or custodian, settle trades, post journal entries or replace an operations analyst.

## 1. The problem

After a trade is executed, an investment-operations analyst checks the firm's internal record against the broker's or custodian's confirmation. The two records should describe the same trade, but they may differ in security ID, date, side, quantity, price, currency, settlement date or account. Some records may also be duplicated or missing. The analyst has to find the right counterpart and work out which differences need attention.

**Who and when:** the main user is a middle- or back-office reconciliation analyst. The check happens when trades and confirmations arrive. Some firms run it intraday; others run an end-of-day batch. This prototype demonstrates a two-CSV batch and does not claim a standard industry schedule.

**Current cost:** the direct baseline is analyst time spent comparing rows, researching breaks, reworking incorrect links and escalating unresolved items. No target firm's volume, exception rate, handling time or loaded labour cost was supplied, so this project claims no cost-per-trade or savings figure. `docs/business_case_discovery.md` provides a measurement worksheet and calculation formula.

**Why start here:** a first version can work from approved CSV exports and confirmation notes without replacing the firm's trading or accounting systems. ML is limited to screening free-text notes. Exact trade fields are checked with rules, and an analyst keeps the final decision.

A plain exact spreadsheet join misses candidate records when source descriptions or identifiers differ. A fuzzy text match can find plausible candidates, but a similar description is not proof that two records represent the same transaction. ReconIQ therefore separates candidate discovery, evidence extraction, deterministic field comparison, narrative exception screening and human review.

**Jurisdiction boundary:** the prototype is not a tax, legal or regulatory rules engine and encodes no jurisdiction-specific legal conclusion. A pilot must select its market, instrument types, identifier standards, settlement calendars / holidays, time zones, broker formats and effective-dated tolerances before use.

### Why this problem fits Tresata

Tresata's public technical note *The Accidental Medallion Architecture* describes staging, profiling, cleaning into a unified schema, semantic joins / record linkage and aggregation, and explicitly names trade reconciliation as a use case. ISDA's *Portfolio Reconciliation in Practice* describes operational issues such as inconsistent account names, wrong-entity booking and incorrect settlement instructions. TCS describes OTC reconciliation exceptions including UTI mismatches, incorrect account data, timing differences and duplicate bookings. These sources establish workflow relevance; this project makes no unsupported estimate of industry-wide costs or savings. See `docs/research_notes.md`.

## 2. How the two research papers inform the design

### Paper A: Han, Wang and Huang (2026), IEEE Access

This paper studies IFRS 16 right-of-use lease-contract intelligence, not trade reconciliation. ReconIQ adapts the architecture pattern of answerability-aware extraction, exact source-span traceability, explicit unknown states and a separate downstream rule layer.

The paper's independent CUAD test set was 68.20% unanswerable. Across five runs it reported HasAns F1 = 21.54 +/- 0.38 and NoAns F1 = 99.34 +/- 0.12. The authors explicitly caution that high aggregate performance mainly reflected conservative no-answer rejection rather than robust answerable-clause extraction. This influenced ReconIQ to preserve `NOT_FOUND` / `AMBIGUOUS`, retain exact source quotes, and avoid treating extraction success as proof that two trades agree.

We do **not** reuse its IFRS 16 D1-D6 rules, CUAD or lease-disclosure datasets, RoBERTa weights, threshold or reported scores as ReconIQ results.

### Paper B: Lippi et al. (2019), CLAUDETTE

CLAUDETTE studies potentially unfair online terms-of-service clauses, not financial trades. It separates two tasks: sentence-level detection of a potentially problematic clause, followed by category classification. Its corpus contains 50 contracts and 12,011 sentences. Its eight legal categories are multi-label: a sentence may belong to more than one category. It compares several approaches, including word / POS TF-IDF SVMs and a category-specific SVM combination (C2), sequence-aware SVM-HMMs and an ensemble (C8). C2 achieved P/R/F1 = 0.798 / 0.782 / 0.781; C8 achieved 0.826 / 0.797 / 0.805 on leave-one-document-out evaluation. For its web service, the authors chose C2 for greater speed despite slightly lower performance.

ReconIQ adapts the **method**, not the legal taxonomy or legal data:

- First detect whether a note segment describes a trade exception.
- Then assign one or more trade-break categories (multi-label, not mutually exclusive).
- Use word and character TF-IDF with a Linear SVM and one-vs-rest category classifiers.
- Include adjacent-sentence context as a simple approximation, while clearly stating that this is not CLAUDETTE's SVM-HMM.
- Maintain explicit annotation guidelines, grouped validation, category-level metrics and false-positive / false-negative review.

We do **not** transfer CLAUDETTE's ToS corpus, “clearly fair / potentially unfair / clearly unfair” legal labels, model scores, C8 ensemble or legal decisions into trade data.

See `docs/research_synthesis.md`, `docs/claudette_method_mapping.md`, `docs/annotation_guidelines.md`, `docs/model_card.md` and `docs/evaluation_report.md` for the full research transfer, category definitions, model limitations and evaluation boundaries.

## 3. What the prototype does

1. Reads an internal OMS / portfolio-accounting CSV and an external broker / custodian CSV.
2. Profiles schemas, missing values and duplicate source trade IDs.
3. Extracts explicitly labelled fields from external `confirmation_text` notes, retaining `ANSWERABLE`, `NOT_FOUND` or `AMBIGUOUS`, exact quote and character offsets.
4. Fills a blank external structured field only when the note contains a single, quote-verifiable value; it does not overwrite a populated field.
5. Generates candidate trade links using exact security IDs and character n-gram TF-IDF similarity over descriptions.
6. Compares six deterministic evidence dimensions R1-R6 using `AGREES`, `CONFLICTS` and `UNKNOWN`.
7. Separately screens note segments with a binary Linear SVM detector, then a multi-label one-vs-rest Linear SVM categoriser.
8. Shows disagreements between ML category suggestions and deterministic rule flags.
9. Leaves conflicts, unknown evidence, duplicates and competing candidate links in the human review queue.

**Critical separation:** the SVM does not determine whether a trade is reconciled. Its outputs are review suggestions. Only the deterministic R1-R6 process can label a pair `MATCHED - EXACT`, and a human remains responsible for any operational decision.

### Exact, probabilistic and human-owned decisions

| Output | Treatment | Source of authority |
|---|---|---|
| Source IDs, source quotes / offsets, date parsing and numeric field checks | Exact / deterministic | Uploaded records and explicit code rules |
| Candidate rank and narrative exception categories | Probabilistic / heuristic | TF-IDF + Linear SVM; margins are not probabilities |
| Market / counterparty tolerances and settlement calendars | Configuration / master data | Must be effective-dated and owned by the target firm's operations / control team |
| Final approval, rejection or investigation | Human decision | Authorised reconciliation analyst |

Schema mappings, tolerances, market calendars and counterparty conventions should live in version-controlled, effective-dated configuration, not model memory. Model weights and thresholds should be versioned with their training-data provenance. The educational prototype does not yet have a persistent rules registry, immutable audit log or enterprise model governance.

### The eight trade exception categories

| Label | Meaning |
|---|---|
| `SECURITY_IDENTITY` | Security / instrument identifier or description differs |
| `TRADE_DATE` | Execution / trade date differs |
| `SIDE` | BUY / SELL direction differs |
| `QUANTITY` | Shares, units or notional differs |
| `PRICE_CURRENCY` | Execution price or currency differs |
| `SETTLEMENT_DATE` | Settlement / value date differs |
| `ACCOUNT_ALLOCATION` | Fund, portfolio, account or counterparty allocation differs |
| `DUPLICATE_OR_MISSING_LINK` | Duplicate, orphan, missing counterpart or competing booking |

A single note can receive multiple labels, e.g. `QUANTITY|PRICE_CURRENCY`.

### R1-R6 deterministic evidence map

| Dimension | Evidence compared |
|---|---|
| R1 Identity | Exact security identifier; description supports candidate discovery only |
| R2 Trade date | Execution date |
| R3 Side | BUY / SELL |
| R4 Quantity | Units / face value / notional |
| R5 Price-currency | Price tolerance and currency code |
| R6 Settlement-allocation | Settlement date, account and counterparty |

`UNKNOWN` is not agreement. A pair is `MATCHED - EXACT` only with an exact security identifier, a unique mutual-best candidate, no duplicate / competing link, and agreement across all six dimensions.

### Why ML is used selectively

Known economic fields are better handled by deterministic rules. The text model is limited to narrative note triage, where wording varies or one segment may mention multiple breaks. The current synthetic test does not prove that ML beats a well-maintained keyword / rules baseline on real broker notes. The next experiment is a same-sample comparison of manual-only, rules-only and hybrid ML; if rules-only performs as well, the simpler design should be preferred.

## 4. Dashboard

The Streamlit app has six tabs:

1. **Reconciliation:** deterministic matches, review queue, analyst decision and CSV export.
2. **Evidence inspector:** original confirmation text, answerability state, exact quote and offsets.
3. **Exception screening:** sentence-level source text, adjacent context, raw SVM detector flag, final flag after narrow post-processing, SVM margin and multi-label categories.
4. **Rule / ML cross-check:** categories found by rules vs. categories suggested by the text model.
5. **Data quality:** required fields, missing values and duplicate IDs.
6. **Research method:** explanation of the two papers, per-category classifier metrics and exact-set error cases, and the limits of each transfer.

A detector margin is an SVM decision score, **not a calibrated probability**. `exception_detected` is a triage flag, not a financial decision.

## 5. Run locally

### Requirements

- Python 3.10+ (Python 3.11 recommended)
- Windows, macOS or Linux
- Internet for initial package installation
- No paid API key required for the default prototype

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

The dashboard opens at a local address, usually `http://localhost:8501`. The app initially uses the bundled synthetic sample files. Upload both internal and external files to test other approved public / synthetic data. It will not mix one custom file with one bundled demo file.

**UI verification note:** the packaging environment did not have Streamlit installed, and network access was unavailable to install it. The Python modules compile and the data / model functions pass the automated tests, but the browser dashboard itself still needs to be launched and inspected locally by the project owner before submission.

### Run the tests and evaluation

```powershell
python -m pytest -q
python scripts/evaluate_synthetic.py
```

Current build results:

- **55 automated tests passed**.
- **Answerability parser fixture:** 23 hand-authored examples: 11 `ANSWERABLE`, 6 `NOT_FOUND`, 6 `AMBIGUOUS`; all 23 expected states and values returned, all 11 answerable quotes verified. This is a rule-parser fixture, not a model benchmark.
- **Exception classifier:** 198 synthetic training rows, including 156 positives and 42 negatives; 30 grouped wording-family validation rows for threshold selection; 50 phrase-family-held-out evaluation rows (42 exceptions and 8 negatives).
- **Synthetic classifier metrics:** raw detection P/R/F1 = 0.9756 / 1.0000 / 0.9877; after the narrow post-processing guard P/R/F1 = 1.0000 / 1.0000 / 1.0000; end-to-end multi-label macro F1 = 0.9218; exact label-set accuracy = 0.90. There were **five** exact label-set errors (`EM0003` to `EM0007`).
- **Trade fixture:** 13 internal and 13 external rows; 3 exact matches, 9 review candidates across four review states, 1 unmatched internal row, 1 unmatched external row and 1 duplicate group.

These results are controlled synthetic software checks, not a finance-expert benchmark, representative corpus, real-world accuracy estimate or production performance claim. The classifier and wording fixtures were designed together. Inspect the per-label support and error cases in `evaluation/exception_classifier_summary.json` and `evaluation/exception_classifier_predictions.csv`. Fifteen representative cases are exported to `examples/prototype_test_cases.csv`, including the five exact label-set failures.

## 6. Input CSV schemas

Both trade files require:

`trade_id, trade_date, security_id, security_description, side, quantity, price, currency`

Optional columns are `settlement_date`, `account` and `counterparty`. The external file can include `confirmation_text`, the note used by evidence extraction and exception screening. See `data/README.md` for the detailed schema and synthetic data notice.

## 7. Known limitations

- The exception model is trained on small, synthetic phrase templates; it is not a finance-domain model and has no independently labelled real trade corpus.
- The 50-row held-out set is grouped by phrase family but is too small and artificial to claim external generalisation.
- Multi-label exact-set errors remain; see the error cases and per-category metrics.
- SVM margins are not probabilities and thresholds are calibrated only on the small synthetic validation subset.
- The local evidence parser recognises explicit labelled values; it is not a general natural-language financial-document model.
- TF-IDF is a lightweight candidate ranker, not a security master or semantic identity system.
- Adjacent sentence context is not full document sequence modelling and does not implement CLAUDETTE's SVM-HMM.
- The app has no production authentication, access control, immutable event store, source hashing, model governance, broker connectivity or settlement / ledger write-back.
- It does not handle partial fills, allocations, novations, lifecycle events, multi-leg OTC products, cash / position reconciliation, corporate actions, fees, FX, valuations, margin, collateral or settlement execution.

## 10. References

1. Tresata Tech Notes, “The Accidental Medallion Architecture” (8 February 2023): https://technotes.tresata.com/medallion-architecture/
2. International Swaps and Derivatives Association (ISDA), *Portfolio Reconciliation in Practice*: https://www.isda.org/a/crpTE/Portfolio-Reconciliation-In-Practice.pdf
3. Tata Consultancy Services, “Automated Reconciliation Process for Over-the-counter Derivatives”: https://www.tcs.com/what-we-do/industries/capital-markets/white-paper/automated-reconciliation-process-otc-markets
4. Han, Y., Wang, Y. and Huang, L. (2026). “An Answerability-Aware Contract Intelligence Framework for Right-of-Use Asset Management With IFRS 16-Oriented Indicator Mapping and Multidimensional Analytics.” *IEEE Access*. DOI: https://doi.org/10.1109/ACCESS.2026.3740089
5. Lippi, M., Pałka, P., Contissa, G., Lagioia, F., Micklitz, H.-W., Sartor, G. and Torroni, P. (2019). “CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service.” *Artificial Intelligence and Law*, 27, 117–139. DOI: https://doi.org/10.1007/s10506-019-09243-2

The supplied legal research papers are methodological references only; their datasets, labels and performance scores are not ReconIQ training data or results.
