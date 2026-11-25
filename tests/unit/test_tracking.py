"""Tests for lineage helpers."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from churnguard.models.tracking import (
    UNKNOWN,
    dvc_data_version,
    file_md5,
    git_commit,
    lineage_tags,
)


def test_file_md5_matches_hashlib(tmp_path: Path) -> None:
    """The streaming hash equals a one-shot MD5 of the same bytes."""
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")

    assert file_md5(path) == hashlib.md5(b"a,b\n1,2\n", usedforsecurity=False).hexdigest()


def test_dvc_data_version_reads_md5_from_pointer_file(tmp_path: Path) -> None:
    """The data version is the md5 that `dvc add` wrote into the .dvc file."""
    pointer = tmp_path / "raw.csv.dvc"
    pointer.write_text("outs:\n- md5: 0123abcd\n  size: 10\n  path: raw.csv\n")

    assert dvc_data_version(pointer) == "0123abcd"


def test_dvc_data_version_untracked_file_is_unknown(tmp_path: Path) -> None:
    """Without a .dvc pointer file the version is reported as unknown."""
    assert dvc_data_version(tmp_path / "missing.csv.dvc") == UNKNOWN


def test_git_commit_outside_repository_is_unknown(tmp_path: Path) -> None:
    """A folder that is not a Git repository has no commit."""
    assert git_commit(tmp_path) == UNKNOWN


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throw-away Git repository with one commit."""
    git = shutil.which("git")
    assert git is not None, "git must be installed to run this test"
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    (tmp_path / "file.txt").write_text("content\n")
    for args in (["init", "-q"], ["add", "file.txt"], [*identity, "commit", "-q", "-m", "init"]):
        subprocess.run([git, *args], cwd=tmp_path, check=True)  # noqa: S603
    return tmp_path


def test_lineage_tags_contains_code_and_data_versions(git_repo: Path) -> None:
    """All four lineage tags are produced, and the commit is a 40-character hash."""
    train_path = git_repo / "train.csv"
    train_path.write_text("x\n1\n")

    tags = lineage_tags(git_repo, git_repo / "raw.csv", train_path)

    assert set(tags) == {"git_commit", "git_dirty", "dvc_data_version", "train_data_md5"}
    assert len(tags["git_commit"]) == 40
    assert tags["git_dirty"] == "false"
