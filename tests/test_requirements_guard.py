"""Ensure a single dependency manifest exists for Streamlit and CI."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_single_requirements_manifest() -> None:
    """Only ``requirements.txt`` may exist as a tracked requirements file."""

    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    tracked_files = result.stdout.splitlines()

    found = {
        Path(name)
        for name in tracked_files
        if Path(name).name.lower().startswith("requirements")
        and Path(name).suffix == ".txt"
    }

    expected = {Path("requirements.txt")}
    missing = sorted(str(path) for path in expected - found)
    unexpected = sorted(str(path) for path in found - expected)

    problems = []
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected: {', '.join(unexpected)}")

    assert not problems, "; ".join(problems)
