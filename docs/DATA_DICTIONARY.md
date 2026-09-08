# Data dictionary

The future master manifest will use one row per original image. Historical
offline augmentations are excluded from the primary experiment.

Planned core identifiers:

| Field | Meaning |
|---|---|
| `original_id` | Persistent source UUID extracted from source metadata or filename. |
| `image_id` | Portable UUID5 derived from source dataset and `original_id`. |
| `sha256` | Identity of the file content. |
| `relative_path` | Portable location relative to the configured data root. |
| `group_id` | Indivisible dependency and split unit. |
| `split` | Canonical train, validation, or test assignment. |

The Gate 2 class order is `Black Rot=0`, `ESCA=1`, `Healthy=2`, and
`Leaf Blight=3`. The master manifest contains 4,062 rows and intentionally
leaves `split` and `split_seed` blank. Its initial `group_id` equals the source
`original_id` and is explicitly provisional until similarity review is closed.
Absolute paths are local convenience fields and may not determine identifiers
or partition assignment.
