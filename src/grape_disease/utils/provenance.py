"""Deterministic source-tree provenance when a Git commit is unavailable."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

TRACKED_ROOTS = ("src", "scripts", "configs", "protocol")
IGNORED_NAMES = {"GATE_APPROVALS.json", "TEST_LOCK.json"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".sqlite", ".sqlite3"}


def source_tree_sha256(repository_root: Path) -> str:
    """Hash executable/configuration inputs using relative paths and bytes."""

    digest = hashlib.sha256()
    files: list[Path] = []
    for root_name in TRACKED_ROOTS:
        root = repository_root / root_name
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(
        files, key=lambda item: item.relative_to(repository_root).as_posix()
    ):
        if path.name in IGNORED_NAMES or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def resolve_code_version(repository_root: Path, configured: str) -> str:
    """Return Git HEAD or a deterministic tree digest for provisional values."""

    if configured and not configured.startswith(("uncommitted", "provisional")):
        return configured
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        if commit:
            return f"git:{commit}"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return f"tree-sha256:{source_tree_sha256(repository_root)}"
