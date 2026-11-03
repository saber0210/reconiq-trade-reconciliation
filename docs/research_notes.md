# Research and real-world problem-discovery notes

This note separates published workflow evidence, supplied-paper findings and ReconIQ's own synthetic prototype results. These are different kinds of evidence and should not be conflated.

## 1. Tresata public technical direction

Tresata Tech Notes, “The Accidental Medallion Architecture” (8 February 2023), describes a data-processing sequence involving staging, profiling, cleaning / unified schema, semantic joins / record linkage, and aggregation. The article explicitly mentions trade reconciliation and tracking customer orders among potential use cases. This is the direct company-fit rationale for selecting trade reconciliation.

Source: https://technotes.tresata.com/medallion-architecture/

## 2. Published portfolio-reconciliation workflow evidence

### ISDA - Portfolio Reconciliation in Practice

ISDA's paper describes reconciliation of trade populations and economic terms between counterparties. Examples include inconsistent account names across trading and operations systems, wrong-entity booking and incorrect settlement instructions, novations not communicated to counterparties, and delayed booking of trade legs. This provides real operational context for account / allocation, duplicate / missing-link and timing exceptions.

Source: https://www.isda.org/a/crpTE/Portfolio-Reconciliation-In-Practice.pdf

### TCS - OTC reconciliation

TCS's capital-markets material identifies exception scenarios such as incorrect account data, UTI mismatches, timing issues and duplicate bookings. It is a vendor publication and is useful as workflow context, not as an independent market-wide statistical study.

Source: https://www.tcs.com/what-we-do/industries/capital-markets/white-paper/automated-reconciliation-process-otc-markets

### Practitioner discussions

Public Reddit discussions in r/CFA and r/hedgefund were considered as qualitative practitioner discovery for spreadsheet-based exception tracking, fixed-income representation and settlement / timing challenges. These are individual anecdotes. They are not representative surveys and are not used to estimate workload, prevalence, market size or savings.

- https://www.reddit.com/r/CFA/comments/m8iio1/how_do_you_handle_tradeposition_reconciliation/
- https://www.reddit.com/r/hedgefund/comments/1nv9a8u/otc_derivatives_ops_how_much_time_goes_to/

## 3. CLAUDETTE paper - what its data actually means

Lippi et al. (2019), “CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service,” *Artificial Intelligence and Law*, 27, 117-139, DOI 10.1007/s10506-019-09243-2.

The source corpus is legal text, not finance data:

- 50 online consumer contracts and 12,011 sentences.
- 1,032 positive sentences (8.6%); after removing fragments shorter than five words, 9,414 sentences remained and 11.0% were positive.
- Eight legal categories: arbitration, unilateral change, content removal, jurisdiction, choice of law, limitation of liability, unilateral termination, contract by using.
- Three annotators jointly wrote guidelines and revised annotations after inspecting errors. A separate ten-contract set was independently annotated by two annotators; Cohen's kappa was 0.871.
- A sentence could have multiple categories; their categorisation task is multi-label, not multi-class.
- C2 category-specific SVM combination: P=.798, R=.782, F1=.781; C8 ensemble: P=.826, R=.797, F1=.805 under leave-one-document-out evaluation. On an additional ten-contract set, ensemble macro P/R/F1 were .782/.708/.736.
- The authors chose C2 for their web service because it was faster despite slightly lower performance.
- Their error analysis identifies context-sensitive false positives / false negatives involving third-party references, liability language and content clauses; they suggest targeted rule-based post-processing and further context modelling.

These details justify borrowing the detection / multi-label classification pattern, annotation discipline, evaluation protocol and error-analysis workflow. They do not justify using the legal corpus, unfairness labels, legal severity gradations, or reported model metrics as trade-reconciliation data.

Full mapping: `claudette_method_mapping.md`.

## 4. Han, Wang and Huang (2026) paper - what its data actually means

The supplied IEEE Access paper studies right-of-use asset management under IFRS 16. It adapts answerability-aware extractive QA, source-span localisation and a separate IFRS 16 indicator-mapping layer. Its CUAD benchmark is a general legal-contract QA dataset, not an IFRS 16-specific or trade dataset. The paper reports that 68.20% of the CUAD independent test instances were unanswerable and that NoAns F1 (99.34) substantially exceeded HasAns F1 (21.54). External enterprise validation also showed much weaker answerable extraction than no-answer rejection.

ReconIQ transfers answerability, exact source quotes, unknown-state preservation and separation of extraction from deterministic rule mapping. It does not use IFRS 16 indicators, CUAD rows, the separate 155-record / 18-field IFRS 16 disclosure dataset, RoBERTa weights, the paper's training loss, threshold or metrics as ReconIQ results.

Full synthesis: `research_synthesis.md`.

