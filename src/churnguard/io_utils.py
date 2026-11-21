"""Small helpers for reading and writing pipeline files.

Keeping file access in one place means every stage handles missing inputs and
missing output folders the same way.

Exported API:
    read_csv: Read a CSV, failing clearly if it does not exist.
    write_csv: Write a CSV, creating the parent folder if needed.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file produced by an earlier pipeline stage.

    Args:
        path: File to read.

    Returns:
        The file contents.

    Raises:
        FileNotFoundError: If the file does not exist, with a hint to run the
            earlier stage first.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}. Run the earlier pipeline stage first."
        )
    return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a data frame to CSV without the index.

    Args:
        frame: Data to write.
        path: Destination file. Its parent folder is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(frame), path)
