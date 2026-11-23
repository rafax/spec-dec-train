"""Access to locally downloaded nvidia/Open-SWE-Traces parquet files."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO = "nvidia/Open-SWE-Traces"
DEFAULT_RAW_DIR = Path("data/raw/open-swe-traces")


@functools.lru_cache(maxsize=4)
def _table(path: str):
    return pq.read_table(path)


def load_row(file: str, row: int, raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """``file`` is the repo-relative parquet path recorded by the survey."""
    return _table(str(raw_dir / file)).slice(row, 1).to_pylist()[0]


def messages_and_tools(row: dict) -> tuple[list[dict], object]:
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    return messages, row.get("tools")


def read_survey(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
