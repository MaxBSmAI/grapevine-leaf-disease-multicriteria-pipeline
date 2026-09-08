"""Conservative related-image candidate generation with human review outputs."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from grape_disease.data.config import SimilarityConfig
from grape_disease.data.fingerprints import (
    cosine_similarity,
    downsampled_rgb_embedding,
    hamming_distance,
    structural_similarity,
)
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.identifiers import make_candidate_pair_id
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

CANDIDATE_FIELDS = (
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "original_id_a",
    "original_id_b",
    "class_name_a",
    "class_name_b",
    "same_class",
    "relative_path_a",
    "relative_path_b",
    "dhash_distance",
    "phash_distance",
    "triggered_by",
    "review_status",
)

METRIC_FIELDS = (
    *CANDIDATE_FIELDS,
    "ssim_uniform_window",
    "rgb_embedding_cosine_similarity",
    "embedding_method",
    "embedding_is_learned",
)

REVIEW_FIELDS = (
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "decision",
    "reason",
    "reviewer",
    "date",
    "evidence",
    "group_id_assigned",
)


@dataclass(frozen=True)
class Candidate:
    """One pair requiring explicit human review."""

    candidate_pair_id: str
    image_id_a: str
    image_id_b: str
    original_id_a: str
    original_id_b: str
    class_name_a: str
    class_name_b: str
    same_class: bool
    relative_path_a: str
    relative_path_b: str
    dhash_distance: int
    phash_distance: int
    triggered_by: str
    review_status: str = "pending_human_review"


class BKTree:
    """Small deterministic BK-tree for hexadecimal perceptual hashes."""

    def __init__(self) -> None:
        self._root: tuple[str, int, dict[int, Any]] | None = None

    def add(self, value: str, index: int) -> None:
        if self._root is None:
            self._root = (value, index, {})
            return
        node = self._root
        while True:
            node_value, _, children = node
            distance = hamming_distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, index, {})
                return
            node = child

    def query(self, value: str, maximum_distance: int) -> list[int]:
        if self._root is None:
            return []
        matches: list[int] = []
        pending = [self._root]
        while pending:
            node_value, index, children = pending.pop()
            distance = hamming_distance(value, node_value)
            if distance <= maximum_distance:
                matches.append(index)
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(
                child
                for edge, child in sorted(children.items(), reverse=True)
                if lower <= edge <= upper
            )
        return sorted(matches)


def _resolve_portable_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Relative path escapes dataset root: {relative_path}")
    return candidate


def generate_hash_candidates(
    frame: pd.DataFrame,
    config: SimilarityConfig,
) -> list[Candidate]:
    """Generate the union of dHash and pHash neighbourhood candidates."""

    dhash_tree = BKTree()
    phash_tree = BKTree()
    candidates: list[Candidate] = []
    for current_index, row in frame.iterrows():
        dhash_value = str(row["dhash"])
        phash_value = str(row["perceptual_hash"])
        previous_indices = set(
            dhash_tree.query(dhash_value, config.dhash_max_distance)
        ) | set(phash_tree.query(phash_value, config.phash_max_distance))
        for previous_index in sorted(previous_indices):
            previous = frame.iloc[previous_index]
            dhash_distance = hamming_distance(dhash_value, str(previous["dhash"]))
            phash_distance = hamming_distance(
                phash_value, str(previous["perceptual_hash"])
            )
            triggers: list[str] = []
            if dhash_distance <= config.dhash_max_distance:
                triggers.append("dhash")
            if phash_distance <= config.phash_max_distance:
                triggers.append("phash")
            if not triggers:
                continue
            image_id_a = str(previous["image_id"])
            image_id_b = str(row["image_id"])
            candidates.append(
                Candidate(
                    candidate_pair_id=make_candidate_pair_id(image_id_a, image_id_b),
                    image_id_a=image_id_a,
                    image_id_b=image_id_b,
                    original_id_a=str(previous["original_id"]),
                    original_id_b=str(row["original_id"]),
                    class_name_a=str(previous["class_name"]),
                    class_name_b=str(row["class_name"]),
                    same_class=str(previous["class_name"]) == str(row["class_name"]),
                    relative_path_a=str(previous["relative_path"]),
                    relative_path_b=str(row["relative_path"]),
                    dhash_distance=dhash_distance,
                    phash_distance=phash_distance,
                    triggered_by="+".join(triggers),
                )
            )
        dhash_tree.add(dhash_value, int(current_index))
        phash_tree.add(phash_value, int(current_index))
    return sorted(
        candidates,
        key=lambda item: (
            min(item.dhash_distance, item.phash_distance),
            item.dhash_distance,
            item.phash_distance,
            item.candidate_pair_id,
        ),
    )


def _measure_candidate(
    candidate: Candidate,
    dataset_root: Path,
    config: SimilarityConfig,
) -> dict[str, Any]:
    path_a = _resolve_portable_path(dataset_root, candidate.relative_path_a)
    path_b = _resolve_portable_path(dataset_root, candidate.relative_path_b)
    with Image.open(path_a) as raw_a, Image.open(path_b) as raw_b:
        image_a = raw_a.convert("RGB")
        image_b = raw_b.convert("RGB")
        ssim = structural_similarity(image_a, image_b, config.ssim_window_size)
        embedding_a = downsampled_rgb_embedding(image_a, config.embedding_size)
        embedding_b = downsampled_rgb_embedding(image_b, config.embedding_size)
        embedding_similarity = cosine_similarity(embedding_a, embedding_b)
    embedding_method = (
        f"standardised_downsampled_rgb_{config.embedding_size}x"
        f"{config.embedding_size}_v1"
    )
    return {
        **asdict(candidate),
        "ssim_uniform_window": round(ssim, 10),
        "rgb_embedding_cosine_similarity": round(embedding_similarity, 10),
        "embedding_method": embedding_method,
        "embedding_is_learned": False,
    }


def _fit_thumbnail(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        color="white",
    )


def create_contact_sheets(
    metrics: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    config: SimilarityConfig,
) -> list[str]:
    """Create paginated side-by-side evidence sheets for human reviewers."""

    if not metrics:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_per_page = config.contact_sheet_pairs_per_page
    page_count = math.ceil(len(metrics) / rows_per_page)
    created: list[str] = []
    thumbnail = config.thumbnail_size
    row_height = thumbnail + 74
    width = thumbnail * 2 + 48
    for page_index in range(page_count):
        page_rows = metrics[
            page_index * rows_per_page : (page_index + 1) * rows_per_page
        ]
        canvas = Image.new("RGB", (width, row_height * len(page_rows) + 36), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            f"Similarity candidates page {page_index + 1}/{page_count}",
            fill="black",
        )
        for row_index, row in enumerate(page_rows):
            y = 36 + row_index * row_height
            path_a = _resolve_portable_path(dataset_root, str(row["relative_path_a"]))
            path_b = _resolve_portable_path(dataset_root, str(row["relative_path_b"]))
            with Image.open(path_a) as raw_a, Image.open(path_b) as raw_b:
                canvas.paste(_fit_thumbnail(raw_a, thumbnail), (12, y))
                canvas.paste(_fit_thumbnail(raw_b, thumbnail), (36 + thumbnail, y))
            label = (
                f"{row['candidate_pair_id']}  dH={row['dhash_distance']}  "
                f"pH={row['phash_distance']}  SSIM={row['ssim_uniform_window']:.4f}  "
                f"RGB-cos={row['rgb_embedding_cosine_similarity']:.4f}"
            )
            draw.text((12, y + thumbnail + 8), label, fill="black")
            draw.text(
                (12, y + thumbnail + 28),
                f"A: {str(row['relative_path_a'])[:88]}",
                fill="black",
            )
            draw.text(
                (12, y + thumbnail + 46),
                f"B: {str(row['relative_path_b'])[:88]}",
                fill="black",
            )
        output_path = output_dir / f"contact_sheet_{page_index + 1:03d}.png"
        canvas.save(output_path, format="PNG", optimize=True)
        created.append(output_path.name)
    return created


def identify_similarity_candidates(
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    config: SimilarityConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate candidates, quantitative evidence, and review templates."""

    root = dataset_root.resolve()
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "original_id",
        "class_name",
        "relative_path",
        "dhash",
        "perceptual_hash",
        "validation_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing similarity fields: {missing}")
    invalid_rows = frame[frame["validation_status"] != "valid"]
    if not invalid_rows.empty:
        invalid_count = len(invalid_rows)
        raise ValueError(
            f"Similarity requires a fully valid manifest; invalid rows: {invalid_count}"
        )
    candidates = generate_hash_candidates(frame, config)
    if dry_run:
        return {
            "dry_run": True,
            "manifest_rows": len(frame),
            "hash_candidates": len(candidates),
        }
    output = ensure_output_directory(output_dir)
    candidate_rows = [asdict(candidate) for candidate in candidates]
    metrics = [_measure_candidate(candidate, root, config) for candidate in candidates]
    write_csv(output / "similarity_candidates.csv", candidate_rows, CANDIDATE_FIELDS)
    write_csv(output / "similarity_metrics.csv", metrics, METRIC_FIELDS)
    review_rows = [
        {
            "candidate_pair_id": candidate.candidate_pair_id,
            "image_id_a": candidate.image_id_a,
            "image_id_b": candidate.image_id_b,
            "decision": "",
            "reason": "",
            "reviewer": "",
            "date": "",
            "evidence": "",
            "group_id_assigned": "",
        }
        for candidate in candidates
    ]
    write_csv(output / "review_decisions_template.csv", review_rows, REVIEW_FIELDS)
    contact_sheets = create_contact_sheets(
        metrics, root, output / "contact_sheets", config
    )
    report = {
        "schema_version": "1.0.0",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(frame),
        "candidate_pairs": len(candidates),
        "same_class_pairs": sum(candidate.same_class for candidate in candidates),
        "cross_class_pairs": sum(not candidate.same_class for candidate in candidates),
        "thresholds": {
            "dhash_max_distance": config.dhash_max_distance,
            "phash_max_distance": config.phash_max_distance,
        },
        "ssim": {
            "implementation": "mean_local_ssim_uniform_window",
            "window_size": config.ssim_window_size,
            "automatic_decision": False,
        },
        "embedding": {
            "implementation": (
                f"standardised_downsampled_rgb_{config.embedding_size}x"
                f"{config.embedding_size}_v1"
            ),
            "learned": False,
            "purpose": "near_duplicate_evidence_only",
            "automatic_decision": False,
        },
        "contact_sheets": contact_sheets,
        "pending_human_review": len(candidates),
        "confirmed_groups_created": False,
        "split_created": False,
        "test_accessed": False,
    }
    write_json(output / "similarity_report.json", report)
    LOGGER.info(
        "Similarity evidence completed", extra={"candidate_pairs": len(candidates)}
    )
    return report
