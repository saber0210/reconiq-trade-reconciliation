# ReconIQ - operational discovery and business-case worksheet

This worksheet is deliberately a blank discovery instrument. The supplied research and public industry sources establish that portfolio / trade reconciliation is a real workflow and identify common exception types; they do **not** provide the target firm's trade volume, break rate, average handling time, loaded labour cost or savings. Fill those only after speaking with an authorised operations owner and observing the actual process.

## 1. Define the workflow being measured

| Discovery field | Ask the process owner | Record |
|---|---|---|
| Team / role | Which middle- or back-office team owns the reconciliation? | `[to validate]` |
| Data population | Which internal system and which broker / custodian confirmations are compared? | `[to validate]` |
| Instruments | Equities, listed derivatives, bonds, OTC derivatives or another class? | `[to validate]` |
| Operating cadence | Per trade, intraday, end-of-day, overnight batch, or another schedule? | `[to validate]` |
| SLA / cutoff | By when must a break be investigated relative to confirmation and settlement cutoffs? | `[to validate]` |
| Exceptions | Which breaks are most frequent and which are most operationally costly? | `[to validate]` |
| Baseline tools | Excel, SQL, a reconciliation product, email queues or manual portal checks? | `[to validate]` |

Do not generalise one firm's cadence or workload to the market. ReconIQ's current demonstration assumes two CSV batches and one-to-one trade comparison.

## 2. Measure the current manual baseline

Collect a representative sample across more than one day, counterparty and instrument type. Avoid selecting only unusually clean or unusually difficult batches.

| Metric | Definition | Value / source |
|---|---|---|
| `N` | Trades received per reconciliation run / day | `[to measure]` |
| `b` | Share of trades requiring manual exception review | `[to measure]` |
| `t` | Average analyst minutes spent on one exception, including research and closure | `[to time]` |
| `r` | Average rework / escalation minutes per exception, measured separately | `[to time]` |
| `c` | Loaded analyst hourly cost, supplied by finance / HR or expressed as an approved range | `[to obtain]` |
| `d` | Operating days per year for the measured process | `[to confirm]` |

A transparent labour baseline is:

```text
Manual review hours per run = N x b x t / 60
Manual review labour cost per run = Manual review hours x c
Annualised labour baseline = Labour cost per run x d
```

Track rework and escalations separately. Track actual late settlement / failed-settlement consequences only when the firm can attribute them reliably. Do not add speculative penalty or loss estimates to the labour baseline.

## 3. Measure whether ReconIQ helps

Run the current process and the prototype / proposed assisted process on the **same approved sample**. Measure:

- analyst minutes per exception before and after assistance;
- proportion of notes correctly surfaced for review;
- false-positive suggestions that create extra review;
- missed exceptions (false negatives), separated by category;
- correct multi-label category sets;
- cases left unresolved because evidence is missing or ambiguous;
- time from batch arrival to closed break;
- analyst override rate and reasons.

A useful candidate comparison is manual-only vs deterministic rules-only vs deterministic rules plus ML narrative triage. The ML design should only be retained if it provides a measured benefit without increasing unacceptable missed-break risk or review burden.

```text
Observed labour time released = (baseline minutes per exception - assisted minutes per exception)
                              x reviewed exception count / 60
Potential gross labour value = Observed labour time released x loaded hourly cost
Net benefit = Potential gross labour value - model, integration, monitoring and control costs
```

This is a measurement framework, not a result. Do not claim labour saved until the same-sample comparison has been completed and reviewed.

## 4. Data, jurisdiction and control questions

Before any pilot, confirm:

1. Which legal entity, market, instrument classes and counterparties are in scope?
2. Which identifiers are authoritative (e.g. ISIN, CUSIP, UTI, internal security master key)?
3. Which trading and settlement calendars, holidays, time zones and market conventions apply?
4. Which price tolerances, quantity units, rounding and currency rules are approved, and who owns them?
5. How are rule changes communicated, effective-dated, tested and versioned?
6. Can a lawful, approved and appropriately de-identified sample be used for model evaluation?
7. Who is allowed to approve or close a break, and what audit evidence must be retained?

The current prototype does not encode jurisdiction-specific legal rules or market-calendar logic. These must be selected and governed for the target market rather than inferred by the SVM or a general language model.

## 5. Evidence to collect before making an investment case

- A process map and system/data lineage for both sides of the reconciliation.
- Representative counts by run, instrument and counterparty.
- A timed baseline sample with exception rate and analyst handling time.
- An adjudicated finance-specific test set with at least two independent annotators.
- Rules-only vs hybrid results on the same held-out sample.
- False-positive / false-negative review with operational owners.
- Security, privacy, audit, retention and integration requirements.

Until these are available, the honest statement is that the current cost is analyst time and operational rework, while the amount is firm-specific and unmeasured in this assignment.
