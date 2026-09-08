"""Re-evaluate a checkpoint on validation only."""

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()
from _entrypoints import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("evaluate_validation", REPOSITORY_ROOT))
