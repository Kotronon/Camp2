#!/usr/bin/env python3
"""
Flag low-quality corrupted NLST cases and optionally filter detections in the
documented corrupted z-region.

This is a simple post-processing mitigation for the robustness report. It does
not improve the model itself; it makes the failure mode explicit and prevents
predictions inside known corrupted regions from being treated as reliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency '{exc.name}' in {sys.executable}.\n"
        "Use the project environment first, for example:\n"
        "  conda activate monai-nlst\n"
        "or run explicitly with:\n"
        f"  /Users/kathi/miniforge3/envs/monai-nlst/bin/python {os.path.basename(__file__)}"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


def load_predictions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_corruption_documentation(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def as_bool(row: dict, key: str, default: bool = False) -> bool:
    value = row.get(key, "")
    if value == "":
        return default
    return str(value).strip().lower() in {"true", "1", "yes"}


def resolve_image_path(image_path: str, image_dir: str) -> Path:
    filename = os.path.basename(image_path)
    candidate = Path(image_dir) / filename
    if candidate.exists():
        return candidate

    path = Path(image_path)
    if path.exists():
        return path

    candidate = PROJECT_ROOT / image_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve image for prediction entry: {image_path}")


def center_voxel_z(box: list[float], image_file: Path) -> float:
    image = nib.load(str(image_file))
    center_world = np.asarray([box[0], box[1], box[2], 1.0], dtype=float)
    center_voxel = np.linalg.inv(image.affine) @ center_world
    return float(center_voxel[2])


def filter_entry(
    entry: dict,
    documentation: dict,
    image_dir: str,
    low_quality_fraction: float,
    filter_corrupted_region: bool,
) -> tuple[dict, dict]:
    filename = os.path.basename(entry["image"])
    doc = documentation.get(filename, {})
    removed_fraction = as_float(doc, "removed_fraction")
    quality_issue_fraction = as_float(doc, "quality_issue_fraction", removed_fraction)
    changed_lung_fraction = as_float(doc, "changed_lung_fraction")
    z_start = as_int(doc, "z_start")
    z_end = as_int(doc, "z_end", -1)
    supports_region_filter = as_bool(doc, "supports_region_filter", default=True)
    low_quality = quality_issue_fraction >= low_quality_fraction

    boxes = entry.get("box", [])
    labels = entry.get("label", [])
    scores = entry.get("label_scores", [])
    keep = [True] * len(boxes)
    removed_in_corrupted_region = 0

    if filter_corrupted_region and supports_region_filter and boxes and z_end >= z_start:
        image_file = resolve_image_path(entry["image"], image_dir)
        for idx, box in enumerate(boxes):
            z_voxel = center_voxel_z(box, image_file)
            if z_start <= z_voxel <= z_end:
                keep[idx] = False
                removed_in_corrupted_region += 1

    mitigated = dict(entry)
    mitigated["box"] = [box for box, should_keep in zip(boxes, keep) if should_keep]
    mitigated["label"] = [label for label, should_keep in zip(labels, keep) if should_keep]
    mitigated["label_scores"] = [score for score, should_keep in zip(scores, keep) if should_keep]
    mitigated["quality_flags"] = {
        "low_quality": low_quality,
        "corruption_type": doc.get("corruption_type", "unknown"),
        "removed_fraction": removed_fraction,
        "quality_issue_fraction": quality_issue_fraction,
        "changed_lung_fraction": changed_lung_fraction,
        "z_start": z_start,
        "z_end": z_end,
        "supports_region_filter": supports_region_filter,
        "removed_predictions_in_corrupted_region": removed_in_corrupted_region,
    }

    summary = {
        "filename": filename,
        "corruption_type": doc.get("corruption_type", "unknown"),
        "low_quality": low_quality,
        "removed_fraction": removed_fraction,
        "quality_issue_fraction": quality_issue_fraction,
        "changed_lung_fraction": changed_lung_fraction,
        "supports_region_filter": supports_region_filter,
        "original_predictions": len(boxes),
        "mitigated_predictions": len(mitigated["box"]),
        "removed_predictions_in_corrupted_region": removed_in_corrupted_region,
    }
    return mitigated, summary


def write_summary(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "corruption_type",
        "low_quality",
        "removed_fraction",
        "quality_issue_fraction",
        "changed_lung_fraction",
        "supports_region_filter",
        "original_predictions",
        "mitigated_predictions",
        "removed_predictions_in_corrupted_region",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mitigate/flag corrupted NLST predictions.")
    parser.add_argument(
        "--predictions",
        default="nlst_detection_outputs/nlst_lung_nodule_corrupted_predictions.json",
        help="Prediction JSON for corrupted images.",
    )
    parser.add_argument(
        "--corruption-documentation",
        default="nlst_detection_outputs/corruption_documentation.csv",
        help="CSV produced by generate_corrupt_data.py.",
    )
    parser.add_argument(
        "--image-dir",
        default="NLST/corrupted_imagesTr",
        help="Directory containing the corrupted images used for inference.",
    )
    parser.add_argument(
        "--output",
        default="nlst_detection_outputs/nlst_lung_nodule_corrupted_predictions_mitigated.json",
        help="Output JSON with quality flags and optional filtering applied.",
    )
    parser.add_argument(
        "--summary-output",
        default="nlst_detection_outputs/mitigation_summary.csv",
        help="CSV summary of quality flags and removed predictions.",
    )
    parser.add_argument(
        "--low-quality-fraction",
        type=float,
        default=0.5,
        help="Mark a case low-quality when removed_fraction is at least this value.",
    )
    parser.add_argument(
        "--no-region-filter",
        action="store_true",
        help="Only add quality flags; do not remove predictions in corrupted z-ranges.",
    )
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    documentation = load_corruption_documentation(args.corruption_documentation)

    mitigated_predictions = []
    summary_rows = []
    for entry in predictions:
        mitigated, summary = filter_entry(
            entry=entry,
            documentation=documentation,
            image_dir=args.image_dir,
            low_quality_fraction=args.low_quality_fraction,
            filter_corrupted_region=not args.no_region_filter,
        )
        mitigated_predictions.append(mitigated)
        summary_rows.append(summary)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(mitigated_predictions, f, indent=2)
    write_summary(summary_rows, args.summary_output)

    low_quality_count = sum(row["low_quality"] for row in summary_rows)
    removed_count = sum(row["removed_predictions_in_corrupted_region"] for row in summary_rows)
    print(f"Processed cases: {len(summary_rows)}")
    print(f"Low-quality cases: {low_quality_count}")
    print(f"Removed predictions in corrupted regions: {removed_count}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
