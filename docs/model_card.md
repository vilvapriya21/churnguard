# Model Card: ChurnGuard churn classifier

## Overview

| | |
|---|---|
| Model | Random forest classifier inside a scikit-learn `Pipeline` (imputation, scaling, one-hot encoding, model) |
| Registry name | `churnguard`, served from the `champion` alias |
| Output | Probability that a customer cancels their plan; flagged as churn at probability ≥ 0.5 |
| Intended users | Retention team, to decide which customers to contact with an offer |
| Trained with | `dvc repro` (stages in `dvc.yaml`, settings in `params.yaml`) |

## Intended use

- **Use for:** ranking current customers by churn risk so the retention team can
  prioritise calls.
- **Do not use for:** pricing, credit or any decision that penalises a
  customer. The model estimates risk; it does not explain why an individual
  customer may leave.

## Data

- **Source:** Telco Customer Churn dataset (Kaggle, `blastchar/telco-customer-churn`),
  7,043 customers, 26.5% churned.
- **Splits** (stratified on `Churn`, seed 42):
  - 20% kept aside as a scoring file that stands in for new customers;
  - the remaining 80% split again into 4,507 training rows and 1,127 test rows.
- **Features:** 19 customer attributes (contract, tenure, charges, services,
  payment method, demographics) plus `avg_monthly_charge` (total charges ÷ tenure).
  `customerID` and `Churn` are never used as features.
- **Cleaning:** 11 blank `TotalCharges` values belong to customers with
  `tenure = 0` and are set to 0.
- **Quality gates:** schema contract, missing values ≤ 5% per column, PSI drift
  between training and scoring data, and leakage checks; see
  `reports/data_quality.json`.

## Model selection

Three candidates were compared with 5-fold stratified cross-validation on the
training rows only. All use the same preprocessing and decision threshold
(0.5); linear and forest models use `class_weight: balanced` so the minority
churn class is not ignored.

| Model | CV ROC-AUC | CV recall | CV precision | CV F1 | CV accuracy |
|---|---|---|---|---|---|
| Logistic regression | 0.843 | 0.802 | 0.516 | 0.628 | 0.748 |
| **Random forest** | **0.844** | **0.792** | **0.526** | **0.632** | **0.755** |
| Gradient boosting | 0.842 | 0.509 | 0.649 | 0.571 | 0.797 |

**Why random forest:** it has the highest cross-validated ROC-AUC, which is the
selection metric, and it keeps recall close to 0.8. The three models rank
customers almost equally well (ROC-AUC within 0.002). Gradient boosting has the
best accuracy but misses about half of the churners (recall 0.51), because it
has no class weighting; for this use, missing a leaving customer costs more
than an unnecessary call. Logistic regression is a close second and would be a
reasonable, more explainable alternative.

## Performance on the held-out test set

1,127 customers never used for training or model selection:

| Metric | Value | Quality bar |
|---|---|---|
| ROC-AUC | 0.849 | ≥ 0.80 ✅ |
| Recall | 0.786 | ≥ 0.70 ✅ |
| Precision | 0.535 | – |
| F1 | 0.637 | – |
| Accuracy | 0.762 | – |

In plain words: the model catches about **4 out of 5** customers who leave.
About **half** of the customers it flags would have stayed, which is the
accepted cost of a retention call.

The quality bar is enforced in code (`src/churnguard/models/register.py`). A
model is registered only if it meets both minimums, and becomes `champion`
only if it also beats the current champion's test ROC-AUC.

## Most important features

From the random forest's feature importances (`reports/figures/feature_importance.png`):

1. Month-to-month contract
2. Tenure (months as a customer)
3. Total charges
4. Two-year contract
5. No online security service
6. No tech support service

Short, flexible contracts and new customers are the strongest churn signals,
consistent with the business intuition.

## Limitations and risks

- **One historical snapshot:** the data is a single public sample. Customer
  behaviour and prices change, so the PSI drift check should be run on every
  new scoring batch, and the model retrained when drift warnings appear.
- **Threshold choice:** 0.5 balances recall and precision for this data.
  Lowering it catches more churners but adds more unnecessary calls; change it
  in `params.yaml` and compare with `dvc metrics diff`.
- **Fairness:** `gender` and `SeniorCitizen` are model inputs. They should be
  monitored so offers are not systematically withheld from a group.
- **Partial input:** the API accepts requests with only some fields; missing
  values are filled with training medians or most frequent values, which makes
  those predictions less reliable.

## Lineage

Every registered version can be traced back to:

- **Code:** the `git_commit` tag on its MLflow run;
- **Data:** the `dvc_data_version` tag (hash of the raw data in its `.dvc`
  pointer file) and `train_data_md5` (hash of the exact training file);
- **Run:** the MLflow run ID, stored as the `source_run_id` tag on the model
  version and in `reports/train_summary.json`.
