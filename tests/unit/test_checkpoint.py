from __future__ import annotations

from pathlib import Path

from grape_disease.training.checkpoint import (
    load_checkpoint,
    save_progress_checkpoint,
)


def test_progress_checkpoint_can_be_atomically_replaced(tmp_path: Path) -> None:
    path = tmp_path / "progress_checkpoint.pth"
    first_digest = save_progress_checkpoint(path, {"epoch": 1})
    assert load_checkpoint(path)["epoch"] == 1
    second_digest = save_progress_checkpoint(path, {"epoch": 2})
    assert load_checkpoint(path)["epoch"] == 2
    assert first_digest != second_digest
    assert path.with_suffix(".pth.sha256").read_text(encoding="utf-8").strip()
