# Text-to-SQL evaluation

Model: `gemini-3.5-flash-lite`. Run on 2026-10-29.

**36 of 37 scored questions passed (97%).** On questions with a data answer: 100% of 26. Database unchanged afterwards: yes. Mean attempts per question: 1.0. Mean time: 3.82 s.

| Category | Passed | Scored | Not scored |
|---|---|---|---|
| simple | 5 | 5 | 0 |
| join_topk | 7 | 7 | 0 |
| dates | 5 | 5 | 0 |
| data_quirks | 5 | 5 | 0 |
| multi_step | 4 | 4 | 0 |
| not_answerable | 2 | 2 | 0 |
| adversarial | 8 | 8 | 0 |
| robustness | 0 | 1 | 0 |

## Failures

- **q37** (robustness): Show me everything in the order_items table.
  - status refused, rows 0: The result would be too large and violate the rule to keep results small using aggregates and LIMIT.
  - no SQL was run
