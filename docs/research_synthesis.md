# Research synthesis: combining answerability-aware extraction and CLAUDETTE-style exception classification

ReconIQ combines complementary ideas from two supplied research papers. Both papers concern legal / contractual text, not securities reconciliation. Their methods are adapted as architecture patterns; their datasets and reported scores are not treated as trade data or ReconIQ results.

## Paper A - Han, Wang and Huang (2026), IEEE Access

The paper studies IFRS 16 right-of-use contract intelligence. Its ROU-TransQA-v2 framework separates answerability classification from extractive span localisation, uses conditional masking so unanswerable examples do not contribute span loss, and passes retained source evidence to a separate expert-informed D1-D6 indicator mapping layer. It preserves yes / no / unknown states.

The paper's CUAD independent-test set was 68.20% unanswerable. Across five runs, it reported HasAns F1 21.54 +/- 0.38 and NoAns F1 99.34 +/- 0.12. This is an important warning: strong rejection of absent evidence did not mean strong extraction of present evidence. The paper also reports external enterprise validation with HasAns F1 19.83 and NoAns F1 97.76, and states that the system does not automate IFRS 16 measurement.

**ReconIQ transfer:** extract structured trade facts only when a confirmation note supports them; retain verbatim quotes and source offsets; preserve `NOT_FOUND` / `AMBIGUOUS`; keep extraction separate from deterministic trade checks; and make uncertainty visible.

**Not transferred:** IFRS 16 D1-D6, CUAD data, the ROU paper's IFRS 16 disclosure dataset, RoBERTa weights, conditional masked loss implementation, model threshold or reported performance values.

## Paper B - Lippi et al. (2019), CLAUDETTE

CLAUDETTE studies potentially unfair clauses in online terms of service. It separates sentence-level detection from category classification. The authors created an annotated corpus of 50 contracts and 12,011 sentences, developed annotation guidelines iteratively, and reported Cohen's kappa 0.871 on a separate ten-contract two-annotator set. It identified eight legal clause categories and allowed a sentence to receive more than one category.

The paper compared lexical SVMs, category-specific SVMs, tree kernels, CNNs, LSTMs, SVM-HMM sequence models and an ensemble. Its C2 category-specific SVM combination achieved P=0.798, R=0.782 and F1=0.781; its C8 ensemble achieved P=0.826, R=0.797 and F1=0.805 on leave-one-document-out evaluation. A ten-contract additional test gave ensemble P=0.782, R=0.708 and F1=0.736. The categorisation task is multi-label and trained on positive clauses only. Table 4 category F1 ranged from .745 for content removal to .970 for jurisdiction. Its error analysis identifies context-sensitive false positives / false negatives (including third-party, liability and content-related sentences) and suggests context or targeted post-processing. The authors use a reviewer-facing interface that highlights sentences and shows category labels; they selected C2 for deployment because it was faster than the ensemble.

**ReconIQ transfer:** first detect exception-bearing note segments, then assign one or more trade-break categories; use category-specific one-vs-rest linear SVMs with word / character TF-IDF; retain sentence context and offsets; create annotation guidance; split validation by phrase family; report category-level metrics; and inspect false positives / false negatives.

**Not transferred:** unfair-clause labels or fairness ratings, the CLAUDETTE ToS corpus, legal decisions, SVM-HMM, the C8 ensemble, or CLAUDETTE's reported performance values.

## Combined architecture

1. **Raw trade and note ingestion:** preserve internal OMS / accounting records and external broker / custodian records.
2. **Answerability-aware extraction:** extract explicit values from confirmation notes with exact source evidence; leave unsupported facts unknown.
3. **Text exception detection:** a binary SVM flags note segments that may describe a trade break.
4. **Multi-label categorisation:** category-specific SVMs suggest one or more of eight trade exception types.
5. **Deterministic rule mapping:** R1-R6 compare security identity, trade date, side, quantity, price/currency and settlement/allocation as `AGREES`, `CONFLICTS` or `UNKNOWN`.
6. **Analyst review and audit output:** expose source text, quote offsets, model suggestions, deterministic conflicts and final human decision.

The ML category layer cannot override rule-based trade status. A suggestion is a request for review, not an approval, booking, settlement instruction or accounting conclusion.

## What is actually built in the prototype

- A local labelled-field parser with `ANSWERABLE`, `NOT_FOUND` and `AMBIGUOUS` states and quote offsets.
- A two-stage synthetic-trained TF-IDF / Linear SVM exception detector and one-vs-rest multi-label categoriser.
- Adjacent-sentence context windows with source offsets. This is not a reproduction of CLAUDETTE's SVM-HMM.
- A narrowly scoped post-processing guard that suppresses a duplicate-link suggestion when a note explicitly confirms a unique one-to-one counterpart.
- A separate deterministic R1-R6 reconciliation engine and human review queue.
- Synthetic held-out and unit tests, with per-category and error-case reports.

## What the prototype results do and do not show

The current generated phrase-family-held-out fixture contains 50 evaluation examples, including 42 exception narratives and 8 clean / negative narratives. On this small synthetic fixture, post-processed detection F1 is 1.000 and end-to-end multi-label macro F1 is 0.9218; exact label-set accuracy is 0.90. There are five multi-label exact-set errors, including missed or extra categories. These scores are **not real-world accuracy**: the corpus is synthetic and small, and some templates share language patterns. They only show that the software executes the expected test flow and expose where category co-occurrence remains weak.

The separate answerability parser fixture has 23 hand-authored examples; all expected statuses and values were returned and all 11 answerable quotes verified. This is a unit-style fixture, not a language-model benchmark.

A finance-specific validation set with expert labels, independent annotators, grouped splits by counterpart / trade batch, and held-out instruments is required before any operational accuracy claim.
