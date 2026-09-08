from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from grape_disease.data.fingerprints import (
    cosine_similarity,
    dhash,
    downsampled_rgb_embedding,
    hamming_distance,
    phash,
    structural_similarity,
)


class FingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        grid = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
        rgb = np.stack((grid, np.flipud(grid), np.fliplr(grid)), axis=-1)
        self.image = Image.fromarray(rgb, mode="RGB")

    def test_hashes_are_deterministic(self) -> None:
        self.assertEqual(dhash(self.image), dhash(self.image.copy()))
        self.assertEqual(phash(self.image), phash(self.image.copy()))

    def test_hamming_distance_is_symmetric(self) -> None:
        self.assertEqual(hamming_distance("0f", "07"), 1)
        self.assertEqual(hamming_distance("07", "0f"), 1)

    def test_identical_images_have_unit_ssim(self) -> None:
        self.assertAlmostEqual(
            structural_similarity(self.image, self.image.copy(), window_size=3),
            1.0,
            places=12,
        )

    def test_identical_embeddings_have_unit_cosine(self) -> None:
        embedding = downsampled_rgb_embedding(self.image, size=16)
        self.assertAlmostEqual(cosine_similarity(embedding, embedding), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
