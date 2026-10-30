# Document Q&A (RAG) evaluation

Model: `gemini-3.8-flash`, embeddings: all-MiniLM-L6-v2, similarity threshold 0.52, split: all. Run on 2026-10-18.

**31 of 35 scored questions passed (89%).** Search found a relevant document in the top passages for 100% of answerable questions. Answers that cited a correct source: 100%.

| Split | Passed | Scored |
|---|---|---|
| dev | 12 | 12 |
| test | 19 | 23 |

| Category | Passed | Scored |
|---|---|---|
| injection | 2 | 2 |
| lookup | 20 | 21 |
| multi_passage | 3 | 3 |
| paraphrase | 0 | 3 |
| unanswerable | 6 | 6 |

## Failures

- **r15** (lookup, test): How long does delivery to Germany take?
  - status no_answer: I don't know based on the documents.
  - answer: I don't know based on the documents.
- **r25** (paraphrase, test): My cushion cover arrived with nothing to stuff it with. Is that normal?
  - status no_answer: I don't know based on the documents.
  - answer: I don't know based on the documents.
- **r26** (paraphrase, test): Do you do next-day delivery?
  - status no_answer: I don't know based on the documents.
  - answer: I don't know based on the documents.
- **r27** (paraphrase, test): Can I buy something without registering an account?
  - status no_answer: I don't know based on the documents.
  - answer: I don't know based on the documents.
