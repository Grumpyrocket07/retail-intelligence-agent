# Retail Intelligence Platform

Analytics on 1M+ retail transactions (UCI Online Retail II): a MySQL data pipeline, churn and sentiment models, a prediction API
and a Text-to-SQL pipeline with layered safety checks. Document Q&A (RAG), a routing agent, observability and a demo interface
are planned (see the roadmap at the bottom).

**Status:** milestone 1 of 8 done (data foundation).

## Milestone 1 - data foundation

Turns the raw UCI **Online Retail II** workbook into a clean, relational MySQL database that every later
milestone (ML features, Text-to-SQL, the agent) builds on.

### Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env           # then edit .env with your MySQL host, port, user, password
```

1. Download Online Retail II from the UCI repository and save the Excel file as
   `data/raw/online_retail_II.xlsx`.
2. Create the database (run `sql/00_create_database.sql` as root in MySQL Workbench or the mysql client).
3. Run the three steps from the project root:

```bash
python -m etl.clean              # ~1 min: cleans the workbook into data/processed/*.csv + audit files
python -m etl.load --reset       # creates the 6 tables and loads them
python -m etl.validate           # 20 checks comparing the database to the cleaning output
```

`etl.validate` exits non-zero if any check fails.

### Schema

| Table | Grain | Notes |
|---|---|---|
| `countries` | one per country | 43 rows |
| `customers` | one per identified customer | country = most frequent country across their invoices |
| `products` | one per stock code | most common cleaned description |
| `orders` | one per invoice | `customer_id` is NULL for anonymous (guest) invoices |
| `order_items` | one per invoice line | holds `unit_price` (prices change over time), `line_total`, `is_outlier` |
| `returns` | one per cancelled line | invoices starting with `C`; quantity stored as a positive number |

### Cleaning rules

| # | Rule | Where |
|---|---|---|
| 1 | Drop the overlap between the two yearly sheets and all exact duplicate rows | `clean.py` |
| 2 | Trim/uppercase stock codes, tidy descriptions, one description per product (most common, ties to most recent) | `normalise`, `build_tables` |
| 3 | Cancellations go to `returns`, not `order_items` | `classify`, `build_tables` |
| 4 | Remove non-product codes (postage, manual adjustments, discounts, bank charges, test entries, etc.) | `NON_PRODUCT_CODES` |
| 5 | Drop zero/negative-price lines and non-cancellation lines with non-positive quantity | `classify` |
| 6 | Keep anonymous invoices (NULL customer) so sales totals stay complete; exclude them from churn later | `orders` |
| 7 | Price stored per line on `order_items`, not on `products` | `build_tables` |
| 8 | Customer country = most frequent; order date = earliest line timestamp | `build_tables` |
| 9 | Flag extreme quantity/price (above the 99.9th percentile of sales lines) with `is_outlier`; do not drop | `build_tables` |

Every removed row is written to `data/audit/dropped_rows.csv` with its reason, and
`data/audit/cleaning_report.json` records what each rule did. A copy of the report from the validated
run is in `docs/cleaning_report.json`.

### Things to know when querying

- `order_items` is **gross sales**. `returns` is separate and cannot be linked to the original sale
  (the source data has no reference), so net figures must be computed as sales minus returns.
- A few extreme rows are real in the source, for example a 80,995-unit order that was cancelled the same
  day. They are kept and flagged with `is_outlier`.
- 7.4% of orders are anonymous; they count toward sales but not toward customer-level analysis.
- The data ends on 9 Dec 2011, so December 2011 is a partial month.

## Data source and attribution

Chen, Daqing. (2019). *Online Retail II*. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/502/online+retail+ii - licensed CC BY 4.0.

## Roadmap

1. Data foundation (done)
2. ML models: churn prediction and sentiment analysis
3. Prediction API
4. Text-to-SQL with validation and injection prevention
5. RAG with ChromaDB
6. Agent routing with LangChain
7. Observability: logging, retries, traces
8. Interface and deployment
