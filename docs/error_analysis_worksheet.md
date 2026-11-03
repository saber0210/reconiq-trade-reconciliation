# ReconIQ classifier error analysis worksheet

A correct binary flag does not mean that the categories are correct. All five rows below were flagged by the raw detector, but the category set was not an exact match.

| Case | Note summary | Expected labels | Predicted labels | Raw detector margin | Error visible in the output |
|---|---|---|---|---:|---|
| EM0003 | OMS says BUY; an external duplicate record says SELL | `DUPLICATE_OR_MISSING_LINK`, `SIDE` | `SIDE` | 0.7595 | Missed `DUPLICATE_OR_MISSING_LINK` |
| EM0004 | Missing counterparty confirmation; security code differs | `DUPLICATE_OR_MISSING_LINK`, `SECURITY_IDENTITY` | No labels | 0.4026 | Missed both categories |
| EM0005 | Broker security code differs; execution date does not align | `SECURITY_IDENTITY`, `TRADE_DATE` | `TRADE_DATE`, `PRICE_CURRENCY` | 0.9435 | Missed `SECURITY_IDENTITY`; added `PRICE_CURRENCY` |
| EM0006 | Portfolio shows 1,500 units; external party shows 1,450; separate account allocated | `QUANTITY`, `ACCOUNT_ALLOCATION` | `ACCOUNT_ALLOCATION` | 1.1415 | Missed `QUANTITY` |
| EM0007 | Settlement value date and price currency differ | `SETTLEMENT_DATE`, `PRICE_CURRENCY` | `SETTLEMENT_DATE`, `PRICE_CURRENCY`, `TRADE_DATE` | 0.2777 | Added `TRADE_DATE` |

### EM0003

- Why do you think the duplicate label was missed?
- Is this an annotation issue, a feature / threshold issue, or a category-boundary issue?
- What would you change, if anything?
- What test would show that the change did not create a new false positive?

### EM0004

- The binary detector flagged the note, but the categoriser returned no labels. What does this tell you about the two-stage design?
- Would you lower a threshold, revise the training examples, or keep the result for human review?
- What is the risk of changing the threshold based on one synthetic example?

### EM0005

- Which words in the note support `SECURITY_IDENTITY`?
- Why might `PRICE_CURRENCY` have been suggested when the note does not mention price or currency?
- What regression example would you add?

### EM0006

- Which phrase signals a quantity break?
- Is the issue likely caused by label co-occurrence or insufficient quantity wording in the synthetic training pool?
- Would you make a code change or add more balanced examples first?

### EM0007

- Does the phrase “value date” explain the extra `TRADE_DATE` label?
- Should trade date and settlement date be distinguished with stronger examples?
- What clean negative example would help test the boundary?

The SVM margin is a decision score, not a calibrated probability. Do not describe 0.9435 as a 94.35% chance of an exception.
