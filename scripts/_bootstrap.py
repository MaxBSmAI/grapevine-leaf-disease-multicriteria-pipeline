"""Local source-tree bootstrap used before the package is installed."""

from __future__ import annotations

import sys
from pathlib import Path


def add_source_tree_to_path() -> Path:
    """Expose the local `src` tree and return the repository root."""

    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return repository_root
