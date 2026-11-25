"""MLflow setup and lineage tags.

*Lineage* means any model can be traced back to exactly what produced it.
Every MLflow run is tagged with:

* ``git_commit``: the code version (and ``git_dirty`` if there were
  uncommitted changes, which makes the commit an incomplete record).
* ``dvc_data_version``: the hash DVC recorded for the raw data file in its
  ``.dvc`` pointer file, i.e. the data version.
* ``train_data_md5``: the hash of the exact training file used, which matches
  the entry for that file in ``dvc.lock``.

Exported API:
    configure_mlflow: Point MLflow at the tracking server and experiment.
    git_commit, git_is_dirty: Code version of the working copy.
    dvc_data_version: Data version from a ``.dvc`` pointer file.
    file_md5: Hash of a file, computed the same way as DVC.
    lineage_tags: All lineage tags for a training run.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Final

import mlflow
import yaml

from churnguard.config import MlflowSettings

UNKNOWN: Final = "unknown"
_HASH_CHUNK_BYTES: Final = 1024 * 1024


def configure_mlflow(settings: MlflowSettings) -> None:
    """Point MLflow at the tracking server and select the experiment.

    Args:
        settings: MLflow section of the environment settings.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)


def _run_git(repo_dir: Path, *args: str) -> str | None:
    """Run a read-only git command and return its output.

    Args:
        repo_dir: Folder inside the Git repository.
        *args: Git arguments, e.g. ``"rev-parse", "HEAD"``.

    Returns:
        The stripped output, or ``None`` if git is unavailable or fails.
    """
    git = shutil.which("git")
    if git is None:
        return None
    # Fixed executable and arguments from this module only; no user input.
    completed = subprocess.run(  # noqa: S603
        [git, *args], cwd=repo_dir, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_commit(repo_dir: Path) -> str:
    """Return the current Git commit ID.

    Args:
        repo_dir: Folder inside the Git repository.

    Returns:
        The full commit hash, or ``"unknown"`` outside a repository.
    """
    return _run_git(repo_dir, "rev-parse", "HEAD") or UNKNOWN


def git_is_dirty(repo_dir: Path) -> bool:
    """Tell whether tracked files have uncommitted changes.

    Args:
        repo_dir: Folder inside the Git repository.

    Returns:
        ``True`` if ``git status`` reports modified tracked files.
    """
    status = _run_git(repo_dir, "status", "--porcelain", "--untracked-files=no")
    return bool(status)


def dvc_data_version(pointer_file: Path) -> str:
    """Read the data hash DVC stored in a ``.dvc`` pointer file.

    Args:
        pointer_file: The ``<data file>.dvc`` file created by ``dvc add``.

    Returns:
        The MD5 hash of the tracked data, or ``"unknown"`` if the file is not
        tracked by DVC yet.
    """
    if not pointer_file.is_file():
        return UNKNOWN
    content = yaml.safe_load(pointer_file.read_text(encoding="utf-8")) or {}
    outputs = content.get("outs") or [{}]
    return str(outputs[0].get("md5", UNKNOWN))


def file_md5(path: Path) -> str:
    """Hash a file with MD5, the same way DVC identifies file versions.

    MD5 is used here as a content fingerprint, not for security.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal MD5 digest.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage_tags(repo_dir: Path, raw_data_path: Path, train_data_path: Path) -> dict[str, str]:
    """Collect the tags that tie a run to its code and data versions.

    Args:
        repo_dir: Folder inside the Git repository.
        raw_data_path: Raw data file; its ``.dvc`` pointer sits next to it.
        train_data_path: The training file used by this run.

    Returns:
        Tag names mapped to values, ready for ``mlflow.set_tags``.
    """
    pointer_file = raw_data_path.with_name(raw_data_path.name + ".dvc")
    return {
        "git_commit": git_commit(repo_dir),
        "git_dirty": str(git_is_dirty(repo_dir)).lower(),
        "dvc_data_version": dvc_data_version(pointer_file),
        "train_data_md5": file_md5(train_data_path),
    }
