"""Measure insertion/deletion faithfulness against random block orders."""

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()
from _entrypoints import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("faithfulness", REPOSITORY_ROOT))
