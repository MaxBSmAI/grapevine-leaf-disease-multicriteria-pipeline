from __future__ import annotations

import csv
import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

from grape_disease.data.config import DatasetConfig
from grape_disease.data.manifest import build_master_manifest, classify_variant


class ManifestTests(unittest.TestCase):
    def _config(self) -> DatasetConfig:
        return DatasetConfig.from_mapping(
            {
                "source_dataset": "fixture",
                "class_to_idx": {"A": 0, "B": 1, "C": 2, "D": 3},
                "expected_total_images": 8,
                "expected_originals": 4,
                "expected_augmented": 4,
                "expected_originals_by_class": {"A": 1, "B": 1, "C": 1, "D": 1},
                "expected_width": 32,
                "expected_height": 32,
                "expected_colour_mode": "RGB",
                "expected_format": "JPEG",
                "image_extensions": [".jpg", ".jpeg"],
                "variant_suffixes": ["flipLR"],
                "require_parquet": False,
            }
        )

    def test_variant_classification(self) -> None:
        config = self._config()
        self.assertEqual(classify_variant(Path("leaf_flipLR.JPG"), config), "flipLR")
        self.assertIsNone(classify_variant(Path("leaf.JPG"), config))

    def test_original_manifest_excludes_offline_variants(self) -> None:
        colours = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (150, 80, 30)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "output"
            for class_name, colour in zip(("A", "B", "C", "D"), colours, strict=True):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                original_id = str(uuid.uuid4())
                image = Image.new("RGB", (32, 32), colour)
                image.save(class_dir / f"{original_id}___leaf.JPG", format="JPEG")
                image.save(
                    class_dir / f"{original_id}___leaf_flipLR.JPG", format="JPEG"
                )
            report = build_master_manifest(root, output, self._config())
            self.assertTrue(report["passed"])
            self.assertEqual(report["originals"], 4)
            self.assertEqual(report["excluded_augmented"], 4)
            with (output / "master_originals_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 4)
            self.assertTrue(
                all(row["is_augmented"] == "False" for row in manifest_rows)
            )
            self.assertTrue(all(row["split"] == "" for row in manifest_rows))
            self.assertTrue(all(row["absolute_path"] == "" for row in manifest_rows))


if __name__ == "__main__":
    unittest.main()
