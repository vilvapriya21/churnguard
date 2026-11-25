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
│   ├── config.py        Typed settings and parameters
│   ├── data/            Column names, ingest, schema contract, drift check
│   ├── features/        Shared feature building (training and API)
│   ├── models/          Train, evaluate, predict, register
│   └── api/             FastAPI application
├── tests/               unit/ and integration/ pytest tests
├── params.yaml          Experiment parameters (seed, split sizes, model settings)
└── pyproject.toml       Dependencies and tool settings; exact versions in uv.lock
```

## Getting started

Requirements: Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/) and Git.

```powershell
# 1. Create .venv and install the exact versions from uv.lock
uv sync

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 3. Install the Git hooks (once per clone)
pre-commit install

# 4. Create your local settings file and fill in any secrets
copy .env.example .env
```

## Dataset

| | |
|---|---|
| Name | Telco Customer Churn |
| Source | <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| File | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Size | 7,043 customers, 21 columns |
| Target | `Churn` (Yes / No), about 26.5% Yes |
| Downloaded | 2026-11-14 |

The file is not stored in Git. Download it from the source above and place it
at the path shown.

### Data handling decisions

- **Blank `TotalCharges`:** 11 rows have a blank instead of a number. All of
  them have `tenure = 0`: brand-new customers who have not paid a bill yet, so
  the blank is filled with `0`. A blank on any customer with `tenure > 0` is
  left missing on purpose, so the data quality checks can catch it.
- **Train / scoring split:** 80% of rows are used for training and 20% are
  kept as a "scoring" file that stands in for new customers. The split is
  stratified on `Churn` and seeded, so it is identical on every run.
- **Train / test split:** the training file is split again (80/20) before any
  preprocessing is fitted, so the test score is an honest estimate.

## Running the pipeline

```powershell
dvc pull     # download the data and model versions recorded in dvc.lock
dvc repro    # run every stage whose inputs changed
```

The six stages are defined in [`dvc.yaml`](dvc.yaml):

| Stage | Reads | Writes |
|---|---|---|
| `ingest` | raw CSV | `data/interim/train.csv`, `scoring.csv` |
| `validate` | interim files | `reports/data_quality.json` (stops the pipeline on failure) |
| `featurize` | interim train file | `data/processed/train.csv`, `test.csv` |
| `train` | processed train file | `models/model.joblib`, `reports/train_summary.json`, MLflow runs |
| `evaluate` | model, processed test file | `reports/metrics.json`, `reports/figures/` |
| `register` | metrics, train summary | `reports/registration.json`, registry aliases |

Each stage can also be run on its own, for example
`python -m churnguard.data.validate`.

Useful commands:

| Command | Shows |
|---|---|
| `dvc dag` | The stages and how they depend on each other |
| `dvc status` | Which stages are out of date |
| `dvc metrics show` | Test scores of the current model |
| `dvc metrics diff` | How test scores changed since the last commit |
| `dvc params diff` | Which parameters changed since the last commit |

## Data versioning: why Git for code and DVC for data

Git is built for small text files: it stores every version of every file, so
committing a dataset or model would make the repository huge and slow, and
GitHub rejects files over 100 MB. DVC keeps the large files in separate
storage (the *remote*) and puts only a small pointer file in Git, holding the
file's hash (its fingerprint). Because the pointer is versioned in Git, every
commit records exactly which data and model belong to that code. `dvc pull`
fetches those files, and `dvc repro` rebuilds anything that is out of date,
so anyone can reproduce the same model from the same commit.

### Storage layout

| Path | Contents | Versioned by |
|---|---|---|
| `data/raw/` | Original download, never edited | DVC (`*.csv.dvc` pointer in Git) |
| `data/interim/` | Cleaned train and scoring files | DVC (pipeline output) |
| `data/processed/` | Model-ready train and test features | DVC (pipeline output) |
| `models/` | Trained model file | DVC (pipeline output) |
| `reports/` | Quality report, metrics, train summary, registration | Git (small JSON); charts in DVC |
| `dvc.yaml`, `dvc.lock`, `params.yaml` | Pipeline definition, exact hashes, parameters | Git |

### Remote

The default remote `localstore` is a folder outside the project
(`../dvc-storage`, next to the assessment folder), configured in
`.dvc/config`. To use cloud storage instead, add it with
`dvc remote add -d <name> s3://<bucket>/<path>` and keep the credentials in
`.env` or `dvc remote modify --local`, which writes to the git-ignored
`.dvc/config.local`. Never commit storage passwords.

## Configuration

| What | Where | Example |
|---|---|---|
| Environment settings (paths, log level, MLflow address) | `configs/dev.yaml`, `staging.yaml`, `prod.yaml` | `log_level: DEBUG` |
| Experiment parameters (seed, split sizes, model settings) | `params.yaml` | `random_seed: 42` |
| Secrets | `.env` (never committed); template in `.env.example` | `MLFLOW_TRACKING_PASSWORD=` |

- Choose the environment with `APP_ENV` (`dev`, `staging` or `prod`; default `dev`).
- Override any setting with an environment variable prefixed `CHURNGUARD_`;
  nested keys use a double underscore, for example
  `CHURNGUARD_MLFLOW__TRACKING_URI=http://localhost:5000`.
- Both files are validated on load, so a typo or wrong type fails immediately.

## Code quality

| Check | Command | What it catches |
|---|---|---|
| Lint | `ruff check src tests` | Style problems, missing docstrings or type hints, unsafe code |
| Format | `ruff format src tests` | Inconsistent formatting (fixes it) |
| Types | `mypy` | Wrong types passed between functions |
| Tests | `pytest` | Broken behaviour; fails below 80% coverage |
| All Git hooks | `pre-commit run --all-files` | Everything above that runs on commit, plus secret scanning |

The pre-commit hooks run ruff, basic file checks, a large-file guard and the
gitleaks secret scanner on every commit.
