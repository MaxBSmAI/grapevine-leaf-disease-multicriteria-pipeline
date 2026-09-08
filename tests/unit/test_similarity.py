from __future__ import annotations

import unittest

import pandas as pd

from grape_disease.data.config import SimilarityConfig
from grape_disease.data.similarity import BKTree, generate_hash_candidates


class SimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimilarityConfig.from_mapping(
            {
                "dhash_max_distance": 1,
                "phash_max_distance": 1,
                "ssim_window_size": 3,
                "embedding_size": 8,
                "contact_sheet_pairs_per_page": 2,
                "thumbnail_size": 32,
            }
        )

    def test_bk_tree_finds_neighbour(self) -> None:
        tree = BKTree()
        tree.add("0000000000000000", 0)
        tree.add("ffffffffffffffff", 1)
        self.assertEqual(tree.query("0000000000000001", 1), [0])

    def test_union_of_hash_candidates_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "image_id": "11111111-1111-4111-8111-111111111111",
                    "original_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "class_name": "A",
                    "relative_path": "A/a.jpg",
                    "dhash": "0000000000000000",
                    "perceptual_hash": "0000000000000000",
                },
                {
                    "image_id": "22222222-2222-4222-8222-222222222222",
                    "original_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                    "class_name": "B",
                    "relative_path": "B/b.jpg",
                    "dhash": "0000000000000001",
                    "perceptual_hash": "ffffffffffffffff",
                },
                {
                    "image_id": "33333333-3333-4333-8333-333333333333",
                    "original_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    "class_name": "C",
                    "relative_path": "C/c.jpg",
                    "dhash": "ffffffffffffffff",
                    "perceptual_hash": "0000000000000001",
                },
            ]
        )
        candidates = generate_hash_candidates(frame, self.config)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.triggered_by for candidate in candidates}, {"dhash", "phash"}
        )


if __name__ == "__main__":
    unittest.main()
