from __future__ import annotations

from pathlib import Path

from grape_disease.utils.provenance import resolve_code_version, source_tree_sha256


def test_source_tree_hash_is_deterministic_and_sensitive(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    file_path = source / "module.py"
    file_path.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    assert first == source_tree_sha256(tmp_path)
    file_path.write_text("VALUE = 2\n", encoding="utf-8")
    assert first != source_tree_sha256(tmp_path)


def test_explicit_code_version_is_preserved(tmp_path: Path) -> None:
    assert resolve_code_version(tmp_path, "published-commit") == "published-commit"
