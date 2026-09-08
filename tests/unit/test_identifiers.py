from __future__ import annotations

import unittest
from pathlib import Path

from grape_disease.utils.identifiers import (
    extract_original_id,
    make_candidate_pair_id,
    make_group_id,
    make_image_id,
)


class IdentifierTests(unittest.TestCase):
    def test_original_uuid_is_extracted_and_canonicalised(self) -> None:
        path = Path("A0B1C2D3-E4F5-4678-9123-ABCDEF012345___leaf.JPG")
        self.assertEqual(
            extract_original_id(path), "a0b1c2d3-e4f5-4678-9123-abcdef012345"
        )

    def test_invalid_filename_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_original_id(Path("not-a-uuid___leaf.JPG"))

    def test_image_id_is_stable_and_path_independent(self) -> None:
        original = "a0b1c2d3-e4f5-4678-9123-abcdef012345"
        first = make_image_id("DataVID - RGB", original)
        second = make_image_id("DataVID - RGB", original.upper())
        self.assertEqual(first, second)

    def test_pair_id_is_order_independent(self) -> None:
        left = "11111111-1111-4111-8111-111111111111"
        right = "22222222-2222-4222-8222-222222222222"
        self.assertEqual(
            make_candidate_pair_id(left, right), make_candidate_pair_id(right, left)
        )

    def test_group_id_is_order_independent_and_singletons_retain_source_id(
        self,
    ) -> None:
        first = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        second = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
        self.assertEqual(make_group_id([first]), first)
        self.assertEqual(
            make_group_id([first, second]), make_group_id([second, first, first])
        )


if __name__ == "__main__":
    unittest.main()
