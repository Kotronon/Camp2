#!/usr/bin/env python3
"""
Create side-by-side PNG evidence for clean vs. corrupted NLST predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from matplotlib.patches import Rectangle
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency '{exc.name}' in {sys.executable}.\n"
        "Use the project environment first, for example:\n"
        "  conda activate monai-nlst\n"
        "or run explicitly with:\n"
        f"  /Users/kathi/miniforge3/envs/monai-nlst/bin/python {os.path.basename(__file__)}"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


def load_predictions(path: str) -> dict[str, dict]:
    with open(path) as f:
        entries = json.load(f)
    return {os.path.basename(entry["image"]): entry for entry in entries}


def load_summary(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def numeric(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def choose_cases(rows: list[dict], max_cases: int) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []

    stable = [
        row
        for row in rows
        if numeric(row, "original_count") > 0
        and numeric(row, "disappeared_count") == 0
        and numeric(row, "new_count") == 0
    ]
    if stable:
        stable.sort(key=lambda row: numeric(row, "mean_matched_iou"), reverse=True)
        selected.append(("stable", stable[0]))

    disappeared = [row for row in rows if numeric(row, "disappeared_count") > 0]
    if disappeared:
        disappeared.sort(key=lambda row: numeric(row, "disappeared_count"), reverse=True)
        selected.append(("disappeared", disappeared[0]))

    new_detection = [row for row in rows if numeric(row, "new_count") > 0]
    if new_detection:
        new_detection.sort(key=lambda row: numeric(row, "new_count"), reverse=True)
        selected.append(("new_detection", new_detection[0]))

    score_drop = [row for row in rows if numeric(row, "mean_score_delta") < -0.02]
    if score_drop:
        score_drop.sort(key=lambda row: numeric(row, "mean_score_delta"))
        selected.append(("score_drop", score_drop[0]))

    return selected[:max_cases]


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
    return 0.0 if union <= 0 else inter_volume / union


def match_boxes(original_boxes: list[list[float]], corrupted_boxes: list[list[float]], iou_threshold: float) -> list[tuple[int, int, float]]:
    if not original_boxes or not corrupted_boxes:
        return []
    original_array = np.asarray(original_boxes, dtype=float)
    corrupted_array = np.asarray(corrupted_boxes, dtype=float)
    candidates = []
    for original_idx, original_box in enumerate(original_array):
        for corrupted_idx, corrupted_box in enumerate(corrupted_array):
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


def resolve_image(filename: str, image_dir: str) -> Path:
    path = Path(image_dir) / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing image: {path}")


def cccwhd_to_world_bounds(box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(box[:3], dtype=float)
    size = np.asarray(box[3:], dtype=float)
    half = size / 2.0
    return center - half, center + half


def world_to_voxel(point: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
    return (np.linalg.inv(affine) @ homogeneous)[:3]


def box_to_display_bounds(box: list[float], image: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    center_voxel = world_to_voxel(np.asarray(box[:3], dtype=float), image.affine)
    spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    spacing = np.where(spacing == 0, 1.0, spacing)

    # The MONAI bundle returns x/y in an LPS-like sign convention for these
    # NLST files, while the displayed array indices are positive voxel indices.
    for axis in (0, 1):
        if center_voxel[axis] < 0 or center_voxel[axis] >= image.shape[axis]:
            center_voxel[axis] = abs(center_voxel[axis])

    size_voxel = np.asarray(box[3:], dtype=float) / spacing
    half_size = size_voxel / 2.0
    voxel_min = center_voxel - half_size
    voxel_max = center_voxel + half_size
    return voxel_min, voxel_max


def filter_boxes(entry: dict, score_threshold: float) -> tuple[list[list[float]], list[float]]:
    boxes = entry.get("box", [])
    scores = entry.get("label_scores", [])
    kept_boxes = []
    kept_scores = []
    for box, score in zip(boxes, scores):
        if score >= score_threshold:
            kept_boxes.append(box)
            kept_scores.append(float(score))
    return kept_boxes, kept_scores


def choose_evidence_box(
    label: str,
    original_entry: dict,
    corrupted_entry: dict,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[str, int] | None:
    original_boxes, original_scores = filter_boxes(original_entry, score_threshold)
    corrupted_boxes, corrupted_scores = filter_boxes(corrupted_entry, score_threshold)
    matches = match_boxes(original_boxes, corrupted_boxes, iou_threshold)
    matched_original = {original_idx for original_idx, _, _ in matches}
    matched_corrupted = {corrupted_idx for _, corrupted_idx, _ in matches}

    if label == "new_detection":
        candidates = [idx for idx in range(len(corrupted_boxes)) if idx not in matched_corrupted]
        if candidates:
            return "corrupted", max(candidates, key=lambda idx: corrupted_scores[idx])

    if label == "disappeared":
        candidates = [idx for idx in range(len(original_boxes)) if idx not in matched_original]
        if candidates:
            return "original", max(candidates, key=lambda idx: original_scores[idx])

    if label == "score_drop" and matches:
        original_idx, corrupted_idx, _ = min(
            matches,
            key=lambda match: corrupted_scores[match[1]] - original_scores[match[0]],
        )
        return "original", original_idx

    if matches:
        original_idx, _, _ = max(matches, key=lambda match: match[2])
        return "original", original_idx
    if original_boxes:
        return "original", int(np.argmax(original_scores))
    if corrupted_boxes:
        return "corrupted", int(np.argmax(corrupted_scores))
    return None


def choose_slice(
    label: str,
    original_entry: dict,
    corrupted_entry: dict,
    image_file: Path,
    score_threshold: float,
    iou_threshold: float,
) -> int:
    image = nib.load(str(image_file))
    evidence = choose_evidence_box(label, original_entry, corrupted_entry, score_threshold, iou_threshold)
    if evidence is None:
        return image.shape[2] // 2
    source, idx = evidence
    boxes, _ = filter_boxes(original_entry if source == "original" else corrupted_entry, score_threshold)
    center = np.asarray(boxes[idx][:3], dtype=float)
    z = int(round(world_to_voxel(center, image.affine)[2]))
    return int(np.clip(z, 0, image.shape[2] - 1))


def draw_boxes(ax, entry: dict, image: nib.Nifti1Image, z_slice: int, score_threshold: float, color: str, prefix: str) -> None:
    boxes, scores = filter_boxes(entry, score_threshold)
    for box, score in zip(boxes, scores):
        voxel_min, voxel_max = box_to_display_bounds(box, image)
        if not (voxel_min[2] <= z_slice <= voxel_max[2]):
            continue
        x = max(voxel_min[0], 0)
        y = max(voxel_min[1], 0)
        width = max(voxel_max[0] - voxel_min[0], 1.0)
        height = max(voxel_max[1] - voxel_min[1], 1.0)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.8))
        ax.plot((voxel_min[0] + voxel_max[0]) / 2.0, (voxel_min[1] + voxel_max[1]) / 2.0, marker="+", color=color)
        ax.text(x, y, f"{prefix} {score:.2f}", color=color, fontsize=8, weight="bold")


def plot_case(
    label: str,
    row: dict,
    original_predictions: dict[str, dict],
    corrupted_predictions: dict[str, dict],
    original_image_dir: str,
    corrupted_image_dir: str,
    output_dir: Path,
    score_threshold: float,
    iou_threshold: float,
) -> Path:
    filename = row["filename"]
    original_file = resolve_image(filename, original_image_dir)
    corrupted_file = resolve_image(filename, corrupted_image_dir)
    original_img = nib.load(str(original_file))
    corrupted_img = nib.load(str(corrupted_file))
    original_data = original_img.get_fdata(dtype=np.float32)
    corrupted_data = corrupted_img.get_fdata(dtype=np.float32)

    original_entry = original_predictions[filename]
    corrupted_entry = corrupted_predictions[filename]
    z_slice = choose_slice(label, original_entry, corrupted_entry, original_file, score_threshold, iou_threshold)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, data, image, entry, title, color in [
        (axes[0], original_data, original_img, original_entry, "clean", "lime"),
        (axes[1], corrupted_data, corrupted_img, corrupted_entry, "corrupted", "red"),
    ]:
        ax.imshow(data[:, :, z_slice].T, cmap="gray", origin="lower", vmin=-1000, vmax=400)
        draw_boxes(ax, entry, image, z_slice, score_threshold, color, "clean" if title == "clean" else "corr")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{label}: {filename} | z={z_slice} | "
        f"orig={row['original_count']} corr={row['corrupted_count']} "
        f"dis={row['disappeared_count']} new={row['new_count']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}_{filename.replace('.nii.gz', '')}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual evidence for NLST robustness cases.")
    parser.add_argument(
        "--summary",
        default="nlst_detection_outputs/original_vs_corrupted_summary.csv",
        help="Case summary CSV from compare_nlst_predictions.py.",
    )
    parser.add_argument(
        "--original",
        default="nlst_detection_outputs/nlst_lung_nodule_predictions.json",
        help="Prediction JSON for clean images.",
    )
    parser.add_argument(
        "--corrupted",
        default="nlst_detection_outputs/nlst_lung_nodule_corrupted_predictions.json",
        help="Prediction JSON for corrupted images.",
    )
    parser.add_argument("--original-image-dir", default="NLST/imagesTr")
    parser.add_argument("--corrupted-image-dir", default="NLST/corrupted_imagesTr")
    parser.add_argument("--output-dir", default="nlst_detection_outputs/visual_evidence")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-cases", type=int, default=4)
    args = parser.parse_args()

    rows = load_summary(args.summary)
    selected = choose_cases(rows, args.max_cases)
    original_predictions = load_predictions(args.original)
    corrupted_predictions = load_predictions(args.corrupted)

    output_paths = []
    for label, row in selected:
        output_paths.append(
            plot_case(
                label=label,
                row=row,
                original_predictions=original_predictions,
                corrupted_predictions=corrupted_predictions,
                original_image_dir=args.original_image_dir,
                corrupted_image_dir=args.corrupted_image_dir,
                output_dir=Path(args.output_dir),
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
        )

    for path in output_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
