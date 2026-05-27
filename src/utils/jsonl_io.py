"""Streaming jsonl read/write for Trajectory objects."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator

from src.labeler.trajectory_schema import Trajectory


def read_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """Lazily yield Trajectory objects from a jsonl file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Trajectory(**json.loads(line))


def write_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> None:
    """Write a sequence of trajectories to a jsonl file (overwrites)."""
    with open(path, "w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def append_trajectory(path: str | Path, t: Trajectory) -> None:
    """Append a single trajectory to a jsonl file (creates if absent)."""
    with open(path, "a") as f:
        f.write(t.model_dump_json() + "\n")
