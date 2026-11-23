## What does this change?

<!-- One or two sentences. Link the task, e.g. "Task 4: data quality checks". -->

## How was it tested?

<!-- Commands you ran and what you saw, e.g. `pytest`, `dvc repro`. -->

## Definition of done

Tick every item that applies. See [docs/definition_of_done.md](../docs/definition_of_done.md).

### Code
- [ ] `ruff check` and `ruff format --check` pass
- [ ] `mypy` passes; every function has type hints
- [ ] Every module, class and public function has a docstring
- [ ] No hard-coded paths, thresholds or credentials
- [ ] No secrets in the diff (gitleaks passes)

### Tests
- [ ] New or changed behaviour has a pytest test
- [ ] All tests pass and coverage is at least 80%

### Data
- [ ] Schema contract, missing-value and drift checks pass (`reports/data_quality.json`)

### Model
- [ ] `dvc repro` runs end to end and `dvc.lock` is committed
- [ ] ROC-AUC ≥ 0.80 and recall ≥ 0.70 on the held-out test set
- [ ] MLflow run is tagged with the Git commit and DVC data version

### Packaging
- [ ] Docker image builds, Trivy finds no fixable CRITICAL issue, SBOM generated
- [ ] Smoke test passes (`/health` and `/predict`)

### Process
- [ ] Branch is short-lived and named `<type>/<short-description>`
- [ ] Commit messages follow Conventional Commits
- [ ] Docs updated where needed (README, docs/, model card)
