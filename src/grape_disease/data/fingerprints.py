"""Deterministic image fingerprints and visual-similarity measurements."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a standard horizontal difference hash as lowercase hexadecimal."""

    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    resized = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    array = np.asarray(resized, dtype=np.uint8)
    bits = array[:, :-1] > array[:, 1:]
    return _bits_to_hex(bits)


@lru_cache(maxsize=8)
def _orthonormal_dct_matrix(size: int) -> npt.NDArray[Any]:
    coordinates = np.arange(size, dtype=np.float64)
    frequencies = coordinates[:, None]
    matrix = np.cos(np.pi * (2.0 * coordinates + 1.0) * frequencies / (2.0 * size))
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return cast(npt.NDArray[Any], matrix)


def phash(
    image: Image.Image, hash_size: int = 8, high_frequency_factor: int = 4
) -> str:
    """Return a DCT perceptual hash without relying on SciPy."""

    if hash_size <= 0 or high_frequency_factor <= 0:
        raise ValueError("Hash dimensions must be positive")
    size = hash_size * high_frequency_factor
    resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float64)
    dct_matrix = _orthonormal_dct_matrix(size)
    transformed = dct_matrix @ array @ dct_matrix.T
    low_frequency = transformed[:hash_size, :hash_size]
    median = np.median(low_frequency.ravel()[1:])
    return _bits_to_hex(low_frequency > median)


def _bits_to_hex(bits: npt.NDArray[Any]) -> str:
    flat = np.asarray(bits, dtype=np.uint8).ravel()
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    width = (flat.size + 3) // 4
    return f"{value:0{width}x}"


def hamming_distance(left: str, right: str) -> int:
    """Return the bit distance between equal-width hexadecimal hashes."""

    if len(left) != len(right):
        raise ValueError("Hashes must have equal hexadecimal width")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def grayscale_array(image: Image.Image) -> npt.NDArray[Any]:
    """Return a grayscale float array in [0, 1]."""

    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def structural_similarity(
    left: Image.Image,
    right: Image.Image,
    window_size: int = 11,
) -> float:
    """Compute mean local SSIM with an unweighted sliding window.

    This implementation follows the standard luminance, contrast, and
    structure formulation using an 11x11 uniform window by default. It is used
    as candidate evidence, never as an automatic duplicate decision.
    """

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and at least 3")
    left_array = grayscale_array(left)
    right_array = grayscale_array(right)
    if left_array.shape != right_array.shape:
        raise ValueError("SSIM inputs must have identical dimensions")
    if min(left_array.shape) < window_size:
        raise ValueError("SSIM window is larger than the input")
    left_windows = np.lib.stride_tricks.sliding_window_view(
        left_array, (window_size, window_size)
    )
    right_windows = np.lib.stride_tricks.sliding_window_view(
        right_array, (window_size, window_size)
    )
    axes = (-2, -1)
    mean_left = left_windows.mean(axis=axes)
    mean_right = right_windows.mean(axis=axes)
    variance_left = left_windows.var(axis=axes, ddof=1)
    variance_right = right_windows.var(axis=axes, ddof=1)
    covariance = (
        (left_windows - mean_left[..., None, None])
        * (right_windows - mean_right[..., None, None])
    ).sum(axis=axes) / (window_size * window_size - 1)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)
    denominator = (mean_left**2 + mean_right**2 + c1) * (
        variance_left + variance_right + c2
    )
    score = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0,
    )
    return float(np.clip(score.mean(), -1.0, 1.0))


def downsampled_rgb_embedding(image: Image.Image, size: int = 32) -> npt.NDArray[Any]:
    """Return a deterministic, non-learned RGB appearance embedding.

    The descriptor standardises each channel after resizing and L2 normalises
    the flattened vector. It is intentionally limited to near-duplicate
    evidence and must not be described as a semantic deep-learning embedding.
    """

    if size < 8:
        raise ValueError("Embedding size must be at least 8")
    resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float64) / 255.0
    mean = array.mean(axis=(0, 1), keepdims=True)
    standard_deviation = array.std(axis=(0, 1), keepdims=True)
    standardised = (array - mean) / np.maximum(standard_deviation, 1e-8)
    vector = standardised.ravel()
    norm = np.linalg.norm(vector)
    return np.asarray(vector / max(float(norm), 1e-12), dtype=np.float64)


def cosine_similarity(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> float:
    """Return cosine similarity for already vectorised descriptors."""

    if left.shape != right.shape:
        raise ValueError("Embedding vectors must have identical shape")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
