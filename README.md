# ChurnGuard

End-to-end customer churn prediction for a telecom company: from a raw CSV to a
containerised prediction API, with data checks, data versioning, experiment
tracking and CI.

- Project charter: [docs/charter.md](docs/charter.md)
- Definition of done: [docs/definition_of_done.md](docs/definition_of_done.md)

## How it fits together

```
Pipeline (DVC + MLflow)
  raw CSV → ingest → validate → featurize → train → evaluate → registry (@champion)

Delivery (GitHub Actions + Docker)
  branch → pull request → CI → Docker image → FastAPI service (/health, /predict)
```

## Repository layout

```
churnguard/
├── .github/workflows/   CI pipeline
├── configs/             Environment settings: dev.yaml, staging.yaml, prod.yaml
├── data/                raw/ interim/ processed/ (tracked by DVC, not Git)
├── docs/                Charter, definition of done, model card
├── models/              Saved model files (tracked by DVC)
├── reports/             Data quality report, metrics, charts
├── src/churnguard/
│   ├── data/            Ingest, schema contract, drift check
│   ├── features/        Shared feature building
│   ├── models/          Train, evaluate, register, predict
│   └── api/             FastAPI application
└── tests/               unit/ and integration/ pytest tests
```

## Dataset

| | |
|---|---|
| Name | Telco Customer Churn |
| Source | <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| File | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Size | 7,043 customers, 21 columns |
| Target | `Churn` (Yes / No) |
| Downloaded | 2026-11-14 |

The file is not stored in Git. Download it from the source above and place it
at the path shown.

## Progress

| Task | Topic | Status |
|---|---|---|
| 1 | Plan the project | Done |
| 2 | Clean Python setup | Not started |
| 3 | Git workflow and CI | Not started |
| 4 | Data quality checks | Not started |
| 5 | Version the data with DVC | Not started |
| 6 | Track experiments with MLflow | Not started |
| 7 | Package it with Docker | Not started |
