from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from grape_disease.data.review_priority import (
    ReviewPriorityConfig,
    prioritise_similarity_review,
)


class ReviewPriorityTests(unittest.TestCase):
    def test_cross_class_is_first_and_no_decisions_are_assigned(self) -> None:
        config = ReviewPriorityConfig.from_mapping(
            {
                "contact_sheet_pairs_per_page": 2,
                "close_hash_distance": 4,
                "high_ssim_threshold": 0.2,
                "high_rgb_cosine_threshold": 0.9,
            }
        )
        frame = pd.DataFrame(
            [
                {
                    "candidate_pair_id": "same-class",
                    "image_id_a": "a",
                    "image_id_b": "b",
                    "class_name_a": "A",
                    "class_name_b": "A",
                    "same_class": "True",
                    "dhash_distance": "2",
                    "phash_distance": "2",
                    "triggered_by": "dhash+phash",
                    "ssim_uniform_window": "0.25",
                    "rgb_embedding_cosine_similarity": "0.95",
                },
                {
                    "candidate_pair_id": "cross-class",
                    "image_id_a": "c",
                    "image_id_b": "d",
                    "class_name_a": "A",
                    "class_name_b": "B",
                    "same_class": "False",
                    "dhash_distance": "20",
                    "phash_distance": "6",
                    "triggered_by": "phash",
                    "ssim_uniform_window": "0.10",
                    "rgb_embedding_cosine_similarity": "0.60",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_path = root / "metrics.csv"
            frame.to_csv(metrics_path, index=False)
            report = prioritise_similarity_review(metrics_path, root / "output", config)
            queue = pd.read_csv(root / "output" / "ai_assisted_review_priorities.csv")
            self.assertEqual(queue.iloc[0]["candidate_pair_id"], "cross-class")
            self.assertEqual(queue.iloc[0]["priority"], "critical")
            self.assertTrue(
                (queue["required_action"] == "independent_human_review").all()
            )
            self.assertEqual(report["scientific_decisions_assigned"], 0)


if __name__ == "__main__":
    unittest.main()
