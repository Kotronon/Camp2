#!/usr/bin/env python3
"""
Compare original and corrupted NLST detection predictions.

The script matches boxes from both prediction JSON files per image and writes a
case-level CSV summary. It is meant for robustness analysis when no nodule
ground-truth boxes are available.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def load_predictions(path: str) -> dict[str, dict]:
    with open(path, "r") as f:
        entries = json.load(f)

    predictions = {}
    for entry in entries:
        key = os.path.basename(entry["image"])
        predictions[key] = entry
    return predictions


def filter_by_score(entry: dict, score_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(entry.get("box", []), dtype=float)
    scores = np.asarray(entry.get("label_scores", []), dtype=float)

    if boxes.size == 0 or scores.size == 0:
        return np.empty((0, 6), dtype=float), np.empty((0,), dtype=float)

    keep = scores >= score_threshold
    return boxes[keep], scores[keep]


def cccwhd_to_xyzxyz(boxes: np.ndarray) -> np.ndarray:
    centers = boxes[:, :3]
    sizes = boxes[:, 3:]
    half_sizes = sizes / 2.0
    return np.concatenate([centers - half_sizes, centers + half_sizes], axis=1)


def box_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = cccwhd_to_xyzxyz(box_a.reshape(1, 6))[0]
    b = cccwhd_to_xyzxyz(box_b.reshape(1, 6))[0]

    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_volume = float(np.prod(inter_size))

    volume_a = float(np.prod(np.maximum(a[3:] - a[:3], 0.0)))
    volume_b = float(np.prod(np.maximum(b[3:] - b[:3], 0.0)))
    union = volume_a + volume_b - inter_volume

    if union <= 0:
        return 0.0
    return inter_volume / union


def match_boxes(
    original_boxes: np.ndarray,
    corrupted_boxes: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for original_idx, original_box in enumerate(original_boxes):
        for corrupted_idx, corrupted_box in enumerate(corrupted_boxes):
            iou = box_iou_3d(original_box, corrupted_box)
            if iou >= iou_threshold:
                candidates.append((iou, original_idx, corrupted_idx))

    candidates.sort(reverse=True)
    used_original = set()
    used_corrupted = set()
    matches = []

    for iou, original_idx, corrupted_idx in candidates:
        if original_idx in used_original or corrupted_idx in used_corrupted:
            continue
        used_original.add(original_idx)
        used_corrupted.add(corrupted_idx)
        matches.append((original_idx, corrupted_idx, iou))

    return matches


def summarize_case(
    filename: str,
    original_entry: dict,
    corrupted_entry: dict,
    score_threshold: float,
    iou_threshold: float,
) -> dict:
    original_boxes, original_scores = filter_by_score(original_entry, score_threshold)
    corrupted_boxes, corrupted_scores = filter_by_score(corrupted_entry, score_threshold)
    matches = match_boxes(original_boxes, corrupted_boxes, iou_threshold)

    score_deltas = [
        corrupted_scores[corrupted_idx] - original_scores[original_idx]
        for original_idx, corrupted_idx, _ in matches
    ]
    matched_ious = [iou for _, _, iou in matches]

    original_count = len(original_boxes)
    corrupted_count = len(corrupted_boxes)
    matched_count = len(matches)
    disappeared_count = original_count - matched_count
    new_count = corrupted_count - matched_count

    return {
        "filename": filename,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "original_count": original_count,
        "corrupted_count": corrupted_count,
        "matched_count": matched_count,
        "disappeared_count": disappeared_count,
        "new_count": new_count,
        "original_max_score": float(original_scores.max()) if original_count else "",
        "corrupted_max_score": float(corrupted_scores.max()) if corrupted_count else "",
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else "",
        "mean_score_delta": float(np.mean(score_deltas)) if score_deltas else "",
    }


def write_summary(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "score_threshold",
        "iou_threshold",
        "original_count",
        "corrupted_count",
        "matched_count",
        "disappeared_count",
        "new_count",
        "original_max_score",
        "corrupted_max_score",
        "mean_matched_iou",
        "mean_score_delta",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original vs. corrupted NLST predictions.")
    parser.add_argument(
        "--original",
        default="nlst_detection_outputs/nlst_lung_nodule_predictions.json",
        help="Prediction JSON for original images.",
    )
    parser.add_argument(
        "--corrupted",
        default="nlst_detection_outputs/nlst_lung_nodule_corrupted_predictions.json",
        help="Prediction JSON for corrupted images.",
    )
    parser.add_argument(
        "--output",
        default="nlst_detection_outputs/original_vs_corrupted_summary.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    original = load_predictions(args.original)
    corrupted = load_predictions(args.corrupted)
    common_files = sorted(set(original) & set(corrupted))

    rows = [
        summarize_case(
            filename,
            original[filename],
            corrupted[filename],
            args.score_threshold,
            args.iou_threshold,
        )
        for filename in common_files
    ]
    write_summary(rows, args.output)

    total_original = sum(row["original_count"] for row in rows)
    total_corrupted = sum(row["corrupted_count"] for row in rows)
    total_matched = sum(row["matched_count"] for row in rows)
    total_disappeared = sum(row["disappeared_count"] for row in rows)
    total_new = sum(row["new_count"] for row in rows)

    print(f"Compared {len(rows)} common cases.")
    print(f"Original detections: {total_original}")
    print(f"Corrupted detections: {total_corrupted}")
    print(f"Matched detections: {total_matched}")
    print(f"Disappeared detections: {total_disappeared}")
    print(f"New detections: {total_new}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
