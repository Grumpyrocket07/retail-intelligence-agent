# Document Q&A (RAG) evaluation

Question set: v2. Model: `gemini-3.8-flash`, embeddings: all-MiniLM-L6-v2, similarity threshold 0.2, split: test. Run on 2026-10-27.

**24 of 26 scored questions passed (92%).** Search found a relevant document in the top passages for 100% of answerable questions. Answers that cited a correct source: 100%.

**2 of 28 questions could not be scored** because the AI service failed. Rerun with `--resume`.

| Split | Passed | Scored |
|---|---|---|
| test | 24 | 26 |

| Category | Passed | Scored |
|---|---|---|
| injection | 1 | 1 |
| lookup | 7 | 7 |
| multi_passage | 5 | 5 |
| paraphrase | 5 | 7 |
| unanswerable | 6 | 6 |

## Failures

- **s16** (paraphrase, test): If I buy something before lunch, how quickly does it leave your warehouse?
  - status no_answer: I don't know based on the documents.
  - answer: I don't know based on the documents.
- **s20** (paraphrase, test): What should I do with a wooden chopping board after washing it?
  - missing one of ['dry']
  - answer: Wooden items should be dried straight away after being wiped with a damp cloth [product-care-and-safety]. They should never be soaked or put in a dishwasher [product-care-and-safety].
