# How we work: branches, commits, reviews and CI

## Trunk-based development

`main` is the trunk and must always work. Nobody pushes to it directly.

1. Update your local `main`: `git switch main` then `git pull`.
2. Create a short-lived branch: `git switch -c feat/data-checks`.
3. Commit small steps on the branch.
4. Push the branch and open a pull request into `main`: `git push -u origin feat/data-checks`.
5. CI runs automatically. A reviewer reads the change and comments.
6. Fix what the reviewer points out, push again, and reply to each comment.
7. When CI is green and the review is approved, merge and delete the branch.

A branch should live for **1–2 days at most**. Long-lived branches drift away
from `main` and end in painful merge conflicts.

### Branch names

`<type>/<short-description>`, using the same types as commit messages:

| Example | For |
|---|---|
| `feat/data-checks` | New functionality |
| `fix/blank-total-charges` | A bug fix |
| `ci/add-docker-job` | CI pipeline changes |
| `docs/model-card` | Documentation only |

## Conventional Commits

Every commit message starts with a type:

```
<type>: <what changed, in the imperative, lower case, no full stop>
```

| Type | When | Example |
|---|---|---|
| `feat` | New behaviour | `feat: add drift check` |
| `fix` | Bug fix | `fix: handle blank TotalCharges` |
| `test` | Tests only | `test: add predict test` |
| `docs` | Documentation only | `docs: update readme` |
| `refactor` | Restructure without changing behaviour | `refactor: centralise column names` |
| `build` | Dependencies, Docker, packaging | `build: add multi-stage dockerfile` |
| `ci` | CI workflow | `ci: add data-check job` |
| `chore` | Anything else (housekeeping) | `chore: create repository layout` |

Why: the history reads like a changelog, and tools can generate release notes
and version numbers from it automatically.

## Code review

As the **author**: keep pull requests small (one task or less), fill in the
template, and make sure CI is green before asking for review.

As the **reviewer**: check that the change does what it says, is tested, and
follows the definition of done. Be specific ("this path is hard-coded; move it
to `configs/`"), not vague ("looks wrong").

## Branch protection on `main`

Set in GitHub under **Settings → Branches → Add branch ruleset** (or *Add
classic branch protection rule*) for the branch `main`:

- [x] Require a pull request before merging (at least 1 approval)
- [x] Require status checks to pass: `lint`, `typecheck`, `test`, `data-check`, `pipeline`, `docker`
- [x] Require branches to be up to date before merging
- [x] Block force pushes
- [x] Restrict deletions

## CI pipeline

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every pull
request into `main` and on every push to `main`:

```
lint ─┐
typecheck ─┼─► data-check ─► pipeline ─► docker
test ─┘
```

| Job | What it runs | Artifacts saved |
|---|---|---|
| `lint` | `ruff check`, `ruff format --check` | none |
| `typecheck` | `mypy` | none |
| `test` | `pytest` with coverage (fails below 80%) | coverage report |
| `data-check` | ingest and the data quality gates | `data_quality.json` |
| `pipeline` | `dvc repro`: full train, evaluate and register | metrics, charts, MLflow store |
| `docker` | build, Trivy scan, SBOM, smoke test, push to GHCR | Trivy report, SBOM |

If any job fails, the jobs after it do not run and the pull request cannot be
merged.
