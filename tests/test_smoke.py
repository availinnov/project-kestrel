"""Smoke tests for the installed command-line interface."""

import subprocess
from importlib.metadata import version


def test_version() -> None:
    result = subprocess.run(
        ["kestrel", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"kestrel {version('project-kestrel')}"
    assert result.stderr == ""
