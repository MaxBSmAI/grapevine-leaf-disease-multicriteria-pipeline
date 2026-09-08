from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.data.group_review import (
    GroupReviewConfig,
    validate_and_consolidate_groups,
)


class GroupReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GroupReviewConfig.from_mapping(
            {
                "protocol_version": "test-v1",
                "allowed_decisions": [
                    "confirmed_same_source",
                    "confirmed_related",
                    "visually_similar_not_related",
                    "uncertain",
                    "rejected",
                ],
                "grouping_decisions": [
                    "confirmed_same_source",
                    "confirmed_related",
                ],
                "blocking_decisions": ["uncertain"],
                "required_metadata_fields": [
                    "reason",
                    "reviewer",
                    "date",
                    "evidence",
                ],
                "require_iso_date": True,
                "block_multiclass_groups": True,
            }
        )

    @staticmethod
    def _manifest(classes: tuple[str, ...]) -> pd.DataFrame:
        originals = (
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        )
        return pd.DataFrame(
            [
                {
                    "image_id": f"image-{index}",
                    "original_id": originals[index],
                    "class_name": class_name,
                    "is_original": "True",
                    "split": "",
                }
                for index, class_name in enumerate(classes)
            ]
        )

    @staticmethod
    def _candidate(
        pair_id: str,
        left: int,
        right: int,
        classes: tuple[str, ...],
    ) -> dict[str, str]:
        return {
            "candidate_pair_id": pair_id,
            "image_id_a": f"image-{left}",
            "image_id_b": f"image-{right}",
            "class_name_a": classes[left],
            "class_name_b": classes[right],
        }

    @staticmethod
    def _decision(
        pair_id: str,
        left: int,
        right: int,
        decision: str,
    ) -> dict[str, str]:
        return {
            "candidate_pair_id": pair_id,
            "image_id_a": f"image-{left}",
            "image_id_b": f"image-{right}",
            "decision": decision,
            "reason": "manual comparison",
            "reviewer": "reviewer-1",
            "date": "2026-07-20",
            "evidence": "contact sheet and metrics",
            "group_id_assigned": "",
        }

    def _run(
        self,
        root: Path,
        manifest: pd.DataFrame,
        candidates: pd.DataFrame,
        decisions: pd.DataFrame,
    ) -> dict[str, Any]:
        manifest_path = root / "manifest.csv"
        candidates_path = root / "candidates.csv"
        decisions_path = root / "decisions.csv"
        output_dir = root / "output"
        manifest.to_csv(manifest_path, index=False)
        candidates.to_csv(candidates_path, index=False)
        decisions.to_csv(decisions_path, index=False)
        return validate_and_consolidate_groups(
            manifest_path=manifest_path,
            candidates_path=candidates_path,
            decisions_path=decisions_path,
            output_dir=output_dir,
            config=self.config,
        )

    def test_blank_review_blocks_group_output_and_writes_report(self) -> None:
        classes = ("A", "A")
        candidates = pd.DataFrame([self._candidate("pair-1", 0, 1, classes)])
        decision = self._decision("pair-1", 0, 1, "")
        decision["reason"] = ""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(
                root, self._manifest(classes), candidates, pd.DataFrame([decision])
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["counts"]["blank_or_invalid_decisions"], 1)
            self.assertFalse((root / "output" / "confirmed_groups.csv").exists())
            report_path = root / "output" / "group_validation_report.json"
            self.assertTrue(report_path.is_file())
            self.assertFalse(json.loads(report_path.read_text())["passed"])

    def test_confirmed_chain_creates_one_deterministic_component(self) -> None:
        classes = ("A", "A", "A")
        candidates = pd.DataFrame(
            [
                self._candidate("pair-1", 0, 1, classes),
                self._candidate("pair-2", 1, 2, classes),
            ]
        )
        decisions = pd.DataFrame(
            [
                self._decision("pair-1", 0, 1, "confirmed_same_source"),
                self._decision("pair-2", 1, 2, "confirmed_related"),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root, self._manifest(classes), candidates, decisions)
            self.assertTrue(report["passed"])
            groups = pd.read_csv(root / "output" / "confirmed_groups.csv")
            self.assertEqual(groups["group_id"].nunique(), 1)
            self.assertEqual(set(groups["group_size"]), {3})
            self.assertFalse(report["split_created"])
            self.assertFalse(report["test_accessed"])

    def test_cross_class_confirmed_component_is_blocked(self) -> None:
        classes = ("A", "B")
        candidates = pd.DataFrame([self._candidate("pair-1", 0, 1, classes)])
        decisions = pd.DataFrame([self._decision("pair-1", 0, 1, "confirmed_related")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root, self._manifest(classes), candidates, decisions)
            self.assertFalse(report["passed"])
            self.assertEqual(report["counts"]["multiclass_groups"], 1)
            self.assertFalse((root / "output" / "confirmed_groups.csv").exists())

    def test_rejected_pair_remains_two_singleton_groups(self) -> None:
        classes = ("A", "A")
        candidates = pd.DataFrame([self._candidate("pair-1", 0, 1, classes)])
        decisions = pd.DataFrame([self._decision("pair-1", 0, 1, "rejected")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._run(root, self._manifest(classes), candidates, decisions)
            self.assertTrue(report["passed"])
            groups = pd.read_csv(root / "output" / "confirmed_groups.csv")
            self.assertEqual(groups["group_id"].nunique(), 2)
            self.assertEqual(set(groups["group_size"]), {1})


if __name__ == "__main__":
    unittest.main()
