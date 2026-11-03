# ReconIQ: beginner's build and demonstration guide

This guide assumes you are new to Python, financial operations and the project. Follow it in order. ReconIQ is an educational prototype; all bundled trades, confirmation notes and classifier examples are synthetic.

## Part 1 - Understand the actual business task

### 1. What is trade reconciliation?

An investment firm records a trade in its internal order-management or portfolio-accounting system. A broker or custodian independently sends a confirmation. An operations analyst checks whether the two records describe the same execution and investigates any break.

Common breaks include a different security ID, wrong BUY/SELL side, quantity or price mismatch, different currency, wrong trade/settlement date, wrong account allocation, duplicate booking or missing confirmation. A wrong match can conceal an operational problem, so a system must be able to say “I don't know” and route a case to a person.

### 2. Why this is a real problem and why Tresata

Tresata's public technical note describes data staging, profiling, standardising, semantic joins / record linkage and aggregation, and explicitly names trade reconciliation as a use case. ISDA and TCS materials describe wrong-entity booking, inconsistent account details, UTI mismatches, timing differences and duplicate bookings as reconciliation problems. The project uses these as workflow context; it does not estimate industry-wide error rates, hours or savings.

### 3. Understand cadence, cost and scope before discussing ROI

The work is triggered for each trade or reconciliation batch. Depending on the firm's feed and control design it may be intraday or end-of-day. Do not claim a universal daily count or review time. Ask the operations owner what systems are compared, when feeds arrive, when breaks must be closed and how many records are reviewed.

The current cost is mainly analyst time: comparing rows, researching exceptions, reworking incorrect links and escalating unresolved items. This project does not have the firm's actual volume, exception rate, handling time or loaded hourly cost. Use `docs/business_case_discovery.md` to measure:

```text
Manual review hours per run = trades per run x exception rate x minutes per exception / 60
Manual review labour cost = manual review hours x loaded hourly cost
```

Measure rework and actual late-settlement impact separately. Do not invent a cost or claim savings from the synthetic demo.

The prototype is jurisdiction-neutral in the sense that it does not interpret tax or legal rules. It does not yet encode market-specific trading / settlement calendars, holidays or counterparty tolerances. A real pilot must choose a market and have its operations owner approve these values.

## Part 2 - Understand the two research contributions

### Paper A: answerability-aware extraction

The supplied Han, Wang and Huang (2026) paper studies IFRS 16 lease-contract intelligence. It is not a trade paper. Its transferable design idea is to separate whether a requested fact is present from extraction of the exact source span, then send retained evidence to a separate expert-defined rule layer.

ReconIQ applies this to broker confirmation notes:

- `ANSWERABLE`: one explicit value is supported by a verbatim source quote.
- `NOT_FOUND`: the note does not state the field; this does not mean false or zero.
- `AMBIGUOUS`: more than one interpretation or value is present.

The exact quote and character offsets stay in the evidence ledger. A blank structured field is filled only when one quote-backed value is found. A populated structured field is not overwritten.

The paper's independent CUAD results are a useful caution: it reported NoAns F1 99.34 and HasAns F1 21.54, with 68.20% of the evaluation examples unanswerable. A high score on rejecting missing answers did not mean strong extraction of present answers. Those are the paper's metrics, not ReconIQ metrics.

### Paper B: CLAUDETTE's detection-then-categorise method

Lippi et al. (2019) studied potentially unfair sentences in online Terms of Service. Their system first detected potentially problematic sentences, then classified those sentences into one or more categories. A single sentence could carry multiple labels. They used category-specific SVM classifiers and word / POS TF-IDF features, compared them with neural and sequence-aware approaches, reviewed false positives and false negatives, and built a reviewer interface that highlighted sentence text and categories.

ReconIQ adapts the structure, not the legal domain:

1. A binary Linear SVM detects whether a note segment describes a potential trade exception.
2. A one-vs-rest multi-label SVM suggests one or more of eight trade-break categories.
3. Adjacent-sentence context is included so a statement can be read with its nearby wording. This is a lightweight approximation, not the paper's SVM-HMM.
4. The screen displays the source sentence, offsets, context, raw SVM flag, post-processing result and suggested labels.

The eight labels describe security identity, trade date, side, quantity, price/currency, settlement date, account/allocation and duplicate/missing link. They are not “fair/unfair” legal grades or severity scores.

The paper's category corpus and metrics are not finance data and are not used as ReconIQ results. The prototype uses a small synthetic training file and phrase-family-held-out synthetic examples.

## Part 3 - Install the software

### 6. Install Python

Install Python 3.11 from https://www.python.org/downloads/ . During Windows installation, enable **Add Python to PATH** if offered. Open PowerShell or Command Prompt and run:

```powershell
py --version
```

### 7. Install Visual Studio Code

Download https://code.visualstudio.com/ . In VS Code choose **File > Open Folder** and select the extracted `ReconIQ_Research_Enhanced` folder.

### 8. Open the terminal

Choose **Terminal > New Terminal**. Make sure the prompt is in the project folder. Verify with:

```powershell
pwd
```

### 9. Create a virtual environment

```powershell
py -m venv .venv
```

A virtual environment keeps this project's packages separate from other Python projects.

### 10. Activate the environment

In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks it, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

When activated, the prompt begins with `(.venv)`.

### 11. Install project libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads Streamlit for the dashboard, pandas / NumPy for data, scikit-learn for TF-IDF and Linear SVM models, python-dotenv for optional local configuration, and pytest for the test suite. Internet is needed for this initial installation. No paid API key is needed.

## Part 4 - Run and explore the dashboard

### 12. Start ReconIQ

```powershell
streamlit run app.py
```

A browser tab should open at a local address, usually `http://localhost:8501`. Keep the terminal running while using the dashboard. The first run trains the small synthetic exception model locally. No model API is called.

The Streamlit browser interface could not be launched in the packaging environment because Streamlit was not installed and network access was unavailable to install it. Running and inspecting all six tabs locally is therefore part of your own pre-submission verification.

### 13. Look at the six tabs

**Reconciliation:** inspect exact matches, review cases, unmatched rows and the analyst decision column. The text classifier cannot change these deterministic statuses.

**Evidence inspector:** choose an external trade, read the original note and inspect extracted fields, answerability state, exact quote and offsets.

**Exception screening:** choose a confirmation note and inspect sentence-level output. Pay attention to `raw_detector_flag`, `exception_detected`, `postprocess_suppressed_labels`, `detection_margin` and `suggested_labels`. The margin is an SVM decision score, not a probability.

**Rule / ML cross-check:** compare deterministic rule categories with classifier suggestions. ML-only suggestions may be false positives; rule-only flags may reflect information in structured fields but not in the note.

**Data quality:** inspect missing input values and duplicate source IDs.

**Research method:** see which parts of both papers were adapted, per-category held-out metrics, the five multi-label exact-set error cases, and which research components were deliberately not implemented.

### 14. Upload custom files safely

Upload both an internal and external CSV. The app will not compare one uploaded file with a bundled demo file. Required columns are:

`trade_id, trade_date, security_id, security_description, side, quantity, price, currency`

Optional columns are `settlement_date`, `account`, `counterparty`; the external CSV may also include `confirmation_text`.

Use synthetic or explicitly approved public data only. Never upload employer, customer, broker, bank or personal financial records.

## Part 5 - Understand the classifier

### 15. Binary detection

The first model estimates whether the segment appears to describe an exception. It is a Linear SVM trained on the synthetic labelled rows.

### 16. Multi-label categorisation

If an exception is detected, eight one-vs-rest Linear SVM classifiers suggest applicable categories. The labels are not mutually exclusive. For example, a note can describe both `QUANTITY` and `PRICE_CURRENCY` mismatches.

### 17. Why TF-IDF and SVM?

CLAUDETTE reported that a category-specific SVM combination (C2) was competitive, while a larger ensemble gave a modest lift but took more runtime; the authors used C2 for their public web service. ReconIQ follows the same simplicity argument: TF-IDF + Linear SVM is small, fast and inspectable for a prototype. This is not a claim that the model performs as well on real trade notes.

The project does not use ML for every task. A spreadsheet / code rule is better for exact dates, quantities, currency and approved tolerances. The SVM is only proposed for narrative notes that use varied wording or mention several breaks together. Whether it is better than a hand-written keyword baseline on real notes has not been proven; compare rules-only and hybrid approaches on the same labelled sample before retaining ML.

### 18. Context and rule post-processing

The model sees a sentence with the adjacent sentence window. This may help when one sentence says “the broker confirmation is…” and the next gives the conflicting field. A narrow post-processing rule suppresses a duplicate/missing-link suggestion only when the note explicitly confirms a unique clean counterpart. The app shows the raw flag and the post-processing result separately.

### 19. Deterministic R1-R6 checks

The reconciliation engine compares:

| Dimension | Check |
|---|---|
| R1 | Security identity |
| R2 | Trade date |
| R3 | BUY / SELL side |
| R4 | Quantity |
| R5 | Price and currency |
| R6 | Settlement date, account and counterparty |

Each dimension is `AGREES`, `CONFLICTS` or `UNKNOWN`. An exact match requires an exact security ID, unique mutual-best candidate, no duplicate / competing link and agreement across all six dimensions. The SVM never overrides this gate.

## Part 6 - Run tests and inspect failure cases

### 20. Run automated tests

Stop the dashboard with `Ctrl+C`, then run:

```powershell
python -m pytest -q
```

The current suite contains **55 tests**. The expected output in the prepared build environment is `55 passed`.

### 21. Run the fixture evaluation

```powershell
python scripts/evaluate_synthetic.py
```

The command regenerates answerability, exception-classifier and trade-reconciliation evaluation outputs under `evaluation/` and example CSVs under `examples/`. The file `examples/prototype_test_cases.csv` contains 15 selected held-out examples, including correct single-label cases, correct multi-label cases and misclassified multi-label cases.

Current synthetic results:

- Answerability: 23 / 23 hand-authored parser states and values returned as expected; 11 / 11 answerable quotes verified.
- Exception model: 198 synthetic training rows, 30 grouped validation rows and 50 phrase-family-held-out evaluation rows.
- Raw detector P/R/F1: 0.9756 / 1.0000 / 0.9877. Post-processed detector: 1.0000 / 1.0000 / 1.0000 on this small fixture.
- End-to-end multi-label macro F1: 0.9218; exact label-set accuracy: 0.90; five exact label-set errors.
- Reconciliation demo: 13 internal / 13 external records, 3 exact matches, 9 review candidates, 1 unmatched on each side and 1 duplicate group.

These are synthetic software checks. The phrase templates and classifier training set were generated together. The 50 held-out cases are not an independently collected finance corpus, and all perfect-looking detector scores must be interpreted with caution. Inspect `evaluation/exception_classifier_predictions.csv` and `evaluation/exception_classifier_summary.json` rather than relying only on the summary values.

## Part 7 - What must happen before any real pilot

- Have at least two investment-operations reviewers independently label a real but approved sample using `docs/annotation_guidelines.md`.
- Report agreement before adjudicating label disagreements.
- Split by trade, source batch, counterparty or institution, not just random sentence row.
- Inspect false-positive and false-negative cases for each category.
- Calibrate thresholds on a validation set; keep a held-out final test.
- Measure binary detection separately from multi-label categorisation and from deterministic reconciliation.
- Keep all ML outputs as analyst suggestions. Do not use them to approve, book or settle a trade.

## Part 8 - Submission preparation

Read `ReconIQ_Submission_Brief.docx` and check every technical statement. The brief is designed as four pages plus a separate one-page AI contribution appendix. Replace every bracketed prompt in the appendix with specific examples from your own work before submitting it. Preserve the actual LLM conversation logs requested by the assignment; the appendix is not a substitute for those logs. Follow `docs/llm_logs_submission_guide.md`. Use `docs/60_minute_owner_plan.md` to prioritise the remaining checks.

Use `docs/assignment_requirements_matrix.md` to see where each assignment requirement is addressed, and `docs/business_case_discovery.md` to gather missing process-volume and cost inputs. A public GitHub repository still needs to be created by you and checked in a private browser window. Do not publish API keys or confidential data.
