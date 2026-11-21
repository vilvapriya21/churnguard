# ChurnGuard Project Charter

## 1. Problem

A telecom company is losing customers to other providers. The retention team
currently has no way to know who is about to leave, so discounts and calls are
offered too late or to the wrong people.

ChurnGuard predicts, for each customer, the probability that they will cancel
their plan ("churn"). The retention team uses the ranked list to contact
high-risk customers before they leave.

## 2. Users of the predictions

| User | How they use ChurnGuard |
|---|---|
| Retention team | Receives a list of customers with a high churn probability and calls them with an offer |
| Retention systems (CRM) | Call `POST /predict` with customer details and receive a churn probability |
| ML engineers | Operate the pipeline, the model registry and the prediction service |

## 3. Success measures

The model does not need to be perfect. Missing a customer who leaves costs more
than calling one who would have stayed, so recall is the primary measure.

| Measure | Target | Where it is checked |
|---|---|---|
| Recall on the churn class (held-out test set) | at least 0.70 | `register.py` quality bar |
| ROC-AUC (held-out test set) | at least 0.80 | `register.py` quality bar |
| Reproducibility | `dvc pull` and `dvc repro` on a fresh clone give the same metrics | Task 5 reproducibility test |
| Traceability | Every registered model links to a Git commit, a DVC data version and an MLflow run | MLflow run tags |
| Test coverage | at least 80% | CI `test` job |

### Out of scope

- Deciding which discount to offer.
- Real-time retraining and live monitoring dashboards (monitoring is described,
  and drift is checked in the pipeline, but no live monitoring service is built).

## 4. Data

- Source: Kaggle "Telco Customer Churn" (`blastchar/telco-customer-churn`).
- One CSV file with 7,043 customers and 21 columns. The target column is `Churn` (Yes / No).
- `customerID` is an identifier and is never used as a feature.
- `TotalCharges` contains a few blank values that are handled during ingest.
- 80% of the rows form the training file. The remaining 20% form a scoring file
  that stands in for "new customers arriving later".

## 5. ML lifecycle and ownership

| # | Step | What happens | Owner | Code |
|---|---|---|---|---|
| 1 | Get data | Read the raw CSV, clean types, split train / scoring | Data engineer | `data/ingest.py` |
| 2 | Check data | Schema contract, missing values, drift | Data engineer | `data/contracts.py`, `data/drift.py` |
| 3 | Prepare features | One shared feature function and preprocessing pipeline | Data scientist | `features/build_features.py` |
| 4 | Train | Fit candidate models, log runs to MLflow | Data scientist | `models/train.py` |
| 5 | Evaluate | Compute metrics and charts on the held-out test set | Data scientist | `models/evaluate.py` |
| 6 | Register model | Apply the quality bar, move aliases challenger → champion | ML engineer | `models/register.py` |
| 7 | Package | Build, scan and publish the Docker image | ML engineer | `Dockerfile`, CI |
| 8 | Serve | FastAPI service loads the champion model | ML engineer | `api/main.py` |
| 9 | Monitor | Drift check on new scoring data, health check on the service | ML engineer | `data/drift.py`, `/health` |
| All | Review | Every change is reviewed in a pull request before merge | Reviewer | GitHub pull requests |

## 6. Lineage

Lineage means any prediction can be traced back to exactly what produced it.
The API serves the model behind the `champion` alias in the MLflow registry.
That model version points to one MLflow run, which stores the parameters and
metrics and is tagged with the Git commit ID (the code version) and the DVC
data version (the hash of the data recorded in `dvc.lock`). Checking out that
commit and running `dvc pull` and `dvc repro` rebuilds the same model.

```
prediction → Docker image tag (Git commit) → model version (@champion)
           → MLflow run → Git commit (code) + DVC hash (data) + params
```

## 7. Environments

| | dev | staging | prod |
|---|---|---|---|
| Where it runs | Developer laptop | GitHub Actions (CI) | Docker container |
| Purpose | Build and experiment | Prove a change is safe to merge | Serve live predictions |
| Data source | Local files pulled with DVC | DVC remote | Live requests to the API |
| Model source | Local MLflow store | MLflow store created in the CI run | MLflow registry, `champion` alias |
| Settings file | `configs/dev.yaml` | `configs/staging.yaml` | `configs/prod.yaml` |
| Log level | DEBUG | INFO | WARNING |
| Secrets | `.env` file (never committed) | GitHub Actions secrets | Environment variables passed at container start |
| Failure behaviour | Fail fast with full detail | Fail the pipeline, block the merge | Return a safe error, never expose internals |

The environment is chosen with the `APP_ENV` variable (`dev` by default).

## 8. Risks

| Risk | Mitigation |
|---|---|
| Bad or changed input data | Pandera contract and missing-value gate stop the pipeline |
| New data differs from training data | PSI drift check: warn at 0.1, fail at 0.25 |
| Leakage inflates scores | `customerID` and `Churn` excluded from features, split before fitting, tested |
| Training and serving compute features differently | One `build_features()` and one saved scikit-learn Pipeline for both |
| A weaker model replaces a good one | Quality bar and champion comparison in `register.py` |
| Secrets leak into Git or the image | `.env` ignored, secret scanner in pre-commit, settings passed as environment variables |
