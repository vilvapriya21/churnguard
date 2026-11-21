# Definition of Done

A change is "done" only when every item that applies to it is ticked. The same
checklist appears in the pull request template, and most items are enforced
automatically by CI.

## Code

- [ ] `ruff check` and `ruff format --check` pass
- [ ] `mypy` passes; every function has type hints
- [ ] Every module, class and public function has a docstring
- [ ] No hard-coded paths, thresholds or credentials; settings come from `configs/`, `params.yaml` or environment variables
- [ ] No secrets in the diff (secret scanner passes)

## Tests

- [ ] New or changed behaviour has a pytest test
- [ ] All tests pass
- [ ] Coverage is at least 80%

## Data

- [ ] Schema contract passes on the training data
- [ ] No column has more than 5% missing values
- [ ] Drift check (PSI) is below the fail limit of 0.25; warnings at 0.1 are explained in the pull request
- [ ] `reports/data_quality.json` is regenerated

## Model

- [ ] `dvc repro` runs end to end and `dvc.lock` is committed
- [ ] ROC-AUC is at least 0.80 and recall is at least 0.70 on the held-out test set
- [ ] The MLflow run is tagged with the Git commit ID and the DVC data version
- [ ] A model is promoted to `champion` only if it beats the current champion

## Packaging

- [ ] The Docker image builds
- [ ] Trivy reports no CRITICAL vulnerability that has a fix
- [ ] The SBOM is generated and saved as a CI artifact
- [ ] The smoke test passes (`/health` and `/predict` answer correctly)

## Process

- [ ] The change is on a short-lived branch and goes through a pull request
- [ ] Commit messages follow Conventional Commits
- [ ] CI is green
- [ ] At least one reviewer approved, and all comments are resolved
- [ ] Documentation is updated (`README.md`, `docs/`, model card) where the change affects it
