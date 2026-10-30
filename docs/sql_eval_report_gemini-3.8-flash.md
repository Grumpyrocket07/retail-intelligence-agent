# Text-to-SQL evaluation

Model: `gemini-3.8-flash`. Run on 2026-10-17.

**37 of 37 scored questions passed (100%).** On questions with a data answer: 100% of 26. Database unchanged afterwards: yes. Mean attempts per question: 1.0. Mean time: 8.49 s.

| Category | Passed | Scored | Not scored |
|---|---|---|---|
| simple | 5 | 5 | 0 |
| join_topk | 7 | 7 | 0 |
| dates | 5 | 5 | 0 |
| data_quirks | 5 | 5 | 0 |
| multi_step | 4 | 4 | 0 |
| not_answerable | 2 | 2 | 0 |
| adversarial | 8 | 8 | 0 |
| robustness | 1 | 1 | 0 |
