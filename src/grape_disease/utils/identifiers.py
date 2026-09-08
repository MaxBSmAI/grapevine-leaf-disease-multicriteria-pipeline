"""Stable identifier construction independent of local absolute paths."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

UUID_PREFIX_RE = re.compile(
    r"^(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})___"
)
IMAGE_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://example.invalid/grape-disease-reproducible/image-id/v1",
)
CANDIDATE_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://example.invalid/grape-disease-reproducible/candidate-id/v1",
)
GROUP_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://example.invalid/grape-disease-reproducible/group-id/v1",
)


def extract_original_id(path: Path) -> str:
    """Extract and canonicalise the source UUID at the beginning of a filename."""

    match = UUID_PREFIX_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Filename does not start with a source UUID: {path.name}")
    return str(uuid.UUID(match.group("uuid")))


def make_image_id(source_dataset: str, original_id: str) -> str:
    """Return a stable UUID5 for a source dataset and source image UUID."""

    canonical_original_id = str(uuid.UUID(original_id))
    name = f"{source_dataset.strip()}:{canonical_original_id}"
    return str(uuid.uuid5(IMAGE_ID_NAMESPACE, name))


def make_candidate_pair_id(image_id_a: str, image_id_b: str) -> str:
    """Return an order-independent stable UUID5 for a pair of image IDs."""

    first, second = sorted((str(uuid.UUID(image_id_a)), str(uuid.UUID(image_id_b))))
    return str(uuid.uuid5(CANDIDATE_ID_NAMESPACE, f"{first}:{second}"))


def make_group_id(original_ids: list[str] | tuple[str, ...] | set[str]) -> str:
    """Return a deterministic group ID for one or more source image UUIDs.

    Singletons retain their source UUID. Multi-image components use UUID5 over
    the sorted, canonical source UUIDs, so row order and local paths cannot
    change the result.
    """

    canonical_ids = sorted({str(uuid.UUID(value)) for value in original_ids})
    if not canonical_ids:
        raise ValueError("At least one original ID is required")
    if len(canonical_ids) == 1:
        return canonical_ids[0]
    return str(uuid.uuid5(GROUP_ID_NAMESPACE, ":".join(canonical_ids)))
