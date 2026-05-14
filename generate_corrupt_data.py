#!/usr/bin/env python3
"""
Generate reproducible NLST corruptions for robustness analysis.

The default run creates 30 comparable cases with unchanged filenames:
10 targeted lung-region occlusions, 10 full-slice dropouts, and 10 low-dose
noise corruptions. This separates a targeted anatomical ablation from more
realistic acquisition-style corruptions.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "NLST" / "imagesTr"
MASK_DIR = PROJECT_ROOT / "NLST" / "masksTr"
CORRUPTED_DATA_DIR = PROJECT_ROOT / "NLST" / "corrupted_imagesTr"
OUTPUT_DIR = PROJECT_ROOT / "nlst_detection_outputs"
DOCUMENTATION_PATH = OUTPUT_DIR / "corruption_documentation.csv"
AIR_HU = -1000.0

DOCUMENTATION_COLUMNS = [
    "filename",
    "corruption_type",
    "z_start",
    "z_end",
    "removed_slices",
    "removed_fraction",
    "changed_voxels",
    "lung_voxels",
    "changed_lung_fraction",
    "noise_sigma_hu",
    "quality_issue_fraction",
    "supports_region_filter",
]


def list_cases(data_dir: Path) -> list[Path]:
    return [Path(path) for path in sorted(glob(str(data_dir / "*.nii.gz")))]


def mask_path_for(case_file: Path) -> Path:
    return MASK_DIR / case_file.name


def axial_region_mask(shape: tuple[int, int, int], z_start: int, z_end: int) -> np.ndarray:
    region_mask = np.zeros(shape, dtype=bool)
    region_mask[:, :, z_start:z_end] = True
    return region_mask


def load_case_and_mask(case_file: Path) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray] | None:
    mask_file = mask_path_for(case_file)
    if not mask_file.exists():
        print(f"Warning: lung mask not found for {case_file.name}; skipping.")
        return None

    img = nib.load(str(case_file))
    data = img.get_fdata(dtype=np.float32)
    lung_mask = nib.load(str(mask_file)).get_fdata() > 0

    if data.shape != lung_mask.shape:
        print(f"Warning: image/mask shape mismatch for {case_file.name}; skipping.")
        return None
    return img, data, lung_mask


def base_row(
    case_file: Path,
    corruption_type: str,
    z_start: int,
    z_end_exclusive: int,
    changed_mask: np.ndarray,
    lung_mask: np.ndarray,
    noise_sigma_hu: float = 0.0,
    quality_issue_fraction: float | None = None,
    supports_region_filter: bool = True,
) -> dict:
    changed_slices = np.any(changed_mask, axis=(0, 1))
    changed_voxels = int(changed_mask.sum())
    lung_voxels = int(lung_mask.sum())
    changed_lung_voxels = int((changed_mask & lung_mask).sum())
    region_slices = max(z_end_exclusive - z_start, 1)
    removed_fraction = float(changed_slices.sum() / region_slices)

    if quality_issue_fraction is None:
        quality_issue_fraction = removed_fraction

    return {
        "filename": case_file.name,
        "corruption_type": corruption_type,
        "z_start": z_start,
        "z_end": z_end_exclusive - 1,
        "removed_slices": int(changed_slices.sum()),
        "removed_fraction": removed_fraction,
        "changed_voxels": changed_voxels,
        "lung_voxels": lung_voxels,
        "changed_lung_fraction": float(changed_lung_voxels / lung_voxels) if lung_voxels else 0.0,
        "noise_sigma_hu": noise_sigma_hu,
        "quality_issue_fraction": quality_issue_fraction,
        "supports_region_filter": supports_region_filter,
    }


def save_corrupted(case_file: Path, img: nib.Nifti1Image, data: np.ndarray, output_dir: Path) -> Path:
    output_file = output_dir / case_file.name
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(output_file))
    return output_file


def corrupt_lung_region_occlusion(case_file: Path, output_dir: Path) -> dict | None:
    loaded = load_case_and_mask(case_file)
    if loaded is None:
        return None
    img, data, lung_mask = loaded

    z_start, z_end = 0, data.shape[2] // 2
    region_mask = axial_region_mask(data.shape, z_start, z_end)
    corruption_mask = lung_mask & region_mask

    if not np.any(corruption_mask):
        print(f"Warning: selected lung region empty for {case_file.name}; skipping.")
        return None

    data[corruption_mask] = AIR_HU
    output_file = save_corrupted(case_file, img, data, output_dir)
    row = base_row(
        case_file=case_file,
        corruption_type="lung_region_occlusion",
        z_start=z_start,
        z_end_exclusive=z_end,
        changed_mask=corruption_mask,
        lung_mask=lung_mask,
        supports_region_filter=True,
    )
    print(f"{case_file.name}: lung_region_occlusion -> {output_file}")
    return row


def corrupt_full_slice_dropout(case_file: Path, output_dir: Path) -> dict | None:
    loaded = load_case_and_mask(case_file)
    if loaded is None:
        return None
    img, data, lung_mask = loaded

    z_start, z_end = data.shape[2] // 4, data.shape[2] // 2
    corruption_mask = axial_region_mask(data.shape, z_start, z_end)
    if not np.any(corruption_mask):
        print(f"Warning: selected slice region empty for {case_file.name}; skipping.")
        return None

    data[corruption_mask] = AIR_HU
    output_file = save_corrupted(case_file, img, data, output_dir)
    row = base_row(
        case_file=case_file,
        corruption_type="full_slice_dropout",
        z_start=z_start,
        z_end_exclusive=z_end,
        changed_mask=corruption_mask,
        lung_mask=lung_mask,
        supports_region_filter=True,
    )
    print(f"{case_file.name}: full_slice_dropout -> {output_file}")
    return row


def corrupt_low_dose_noise(
    case_file: Path,
    output_dir: Path,
    rng: np.random.Generator,
    noise_sigma_hu: float,
) -> dict | None:
    loaded = load_case_and_mask(case_file)
    if loaded is None:
        return None
    img, data, lung_mask = loaded

    body_mask = data > -950
    if not np.any(body_mask):
        print(f"Warning: body mask empty for {case_file.name}; skipping.")
        return None

    noise = rng.normal(loc=0.0, scale=noise_sigma_hu, size=int(body_mask.sum())).astype(np.float32)
    data[body_mask] = np.clip(data[body_mask] + noise, -1024.0, 3071.0)
    output_file = save_corrupted(case_file, img, data, output_dir)
    row = base_row(
        case_file=case_file,
        corruption_type="low_dose_noise",
        z_start=0,
        z_end_exclusive=data.shape[2],
        changed_mask=body_mask,
        lung_mask=lung_mask,
        noise_sigma_hu=noise_sigma_hu,
        quality_issue_fraction=min(noise_sigma_hu / 150.0, 1.0),
        supports_region_filter=False,
    )
    row["removed_slices"] = 0
    row["removed_fraction"] = 0.0
    print(f"{case_file.name}: low_dose_noise sigma={noise_sigma_hu:g}HU -> {output_file}")
    return row


def clear_nii_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.nii.gz"):
        path.unlink()


def generate_default_protocol(
    cases_per_type: int,
    output_dir: Path,
    documentation_path: Path,
    clear_output: bool,
    noise_sigma_hu: float,
    seed: int,
) -> pd.DataFrame:
    if clear_output:
        clear_nii_outputs(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = list_cases(DATA_DIR)
    protocol = [
        ("lung_region_occlusion", 0),
        ("full_slice_dropout", cases_per_type),
        ("low_dose_noise", cases_per_type * 2),
    ]

    rows = []
    for corruption_type, start_idx in protocol:
        selected_cases = cases[start_idx : start_idx + cases_per_type]
        for case_offset, case_file in enumerate(selected_cases):
            if corruption_type == "lung_region_occlusion":
                row = corrupt_lung_region_occlusion(case_file, output_dir)
            elif corruption_type == "full_slice_dropout":
                row = corrupt_full_slice_dropout(case_file, output_dir)
            elif corruption_type == "low_dose_noise":
                rng = np.random.default_rng(seed + start_idx + case_offset)
                row = corrupt_low_dose_noise(case_file, output_dir, rng, noise_sigma_hu)
            else:
                raise ValueError(f"Unsupported corruption_type: {corruption_type}")
            if row is not None:
                rows.append(row)

    documentation = pd.DataFrame(rows, columns=DOCUMENTATION_COLUMNS)
    documentation_path.parent.mkdir(parents=True, exist_ok=True)
    documentation.to_csv(documentation_path, index=False)
    print(f"Wrote documentation: {documentation_path}")
    return documentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLST robustness corruptions.")
    parser.add_argument(
        "--cases-per-type",
        type=int,
        default=10,
        help="Number of cases for each corruption type. Default creates 30 total cases.",
    )
    parser.add_argument(
        "--noise-sigma-hu",
        type=float,
        default=75.0,
        help="Gaussian HU noise sigma for low_dose_noise cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for low_dose_noise.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CORRUPTED_DATA_DIR),
        help="Folder for corrupted NIfTI files.",
    )
    parser.add_argument(
        "--documentation",
        default=str(DOCUMENTATION_PATH),
        help="CSV path for corruption metadata.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove existing .nii.gz files from the corrupted output folder first.",
    )
    args = parser.parse_args()

    generate_default_protocol(
        cases_per_type=args.cases_per_type,
        output_dir=Path(args.output_dir),
        documentation_path=Path(args.documentation),
        clear_output=not args.keep_existing,
        noise_sigma_hu=args.noise_sigma_hu,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
