#!/usr/bin/env python3
"""
Einstiegsskript fuer NLST Detection mit MONAI.

Ohne Argumente prueft das Skript ein NLST-Volume visuell. Mit --run-bundle
startet es das MONAI Model-Zoo Bundle lung_nodule_ct_detection.
"""

import argparse
import glob
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "NLST", "corrupted_imagesTr")
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "model-zoo", "models", "lung_nodule_ct_detection")
CHECKPOINT = os.path.join(BUNDLE_DIR, "models", "model.pt")
CPU_CHECKPOINT = os.path.join(BUNDLE_DIR, "models", "model_cpu.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "nlst_detection_outputs")


def ensure_cpu_checkpoint_if_needed() -> None:
    import torch

    if torch.cuda.is_available():
        return

    if os.path.exists(CPU_CHECKPOINT):
        return

    print("CUDA is not available. Creating CPU-compatible checkpoint model_cpu.pt ...")
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"), weights_only=True)
    torch.save(checkpoint, CPU_CHECKPOINT)


def run_bundle_inference(max_cases: int, start_case: int) -> None:
    if not os.path.exists(BUNDLE_DIR):
        raise FileNotFoundError(f"MONAI Bundle not found: {BUNDLE_DIR}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Pretrained Checkpoint missing: {CHECKPOINT}")

    ensure_cpu_checkpoint_if_needed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config_arg = "['configs/inference.json','../../../nlst_bundle_inference.json']"
    command = [
        sys.executable,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        config_arg,
        "--nlst_max_cases",
        str(max_cases),
        "--nlst_start_case",
        str(start_case),
    ]

    print("Starting MONAI Model-Zoo Inference:")
    print(" ".join(command))
    subprocess.run(command, cwd=BUNDLE_DIR, check=True)


def preview_first_volume() -> None:
    import matplotlib.pyplot as plt
    from monai.data import DataLoader, Dataset
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
    )
    from monai.utils import set_determinism

    set_determinism(seed=0)

    if not os.path.exists(DATA_DIR):
        print("Please download the NLST dataset and extract it to NLST/imagesTr.")
        print("The folder should contain e.g., NLST/imagesTr/NLST_0001_0000.nii.gz.")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))
    if not image_files:
        raise FileNotFoundError(f"No .nii.gz files found in folder: {DATA_DIR}")

    sample_file = image_files[0]
    print(f"Loading first NLST file: {sample_file}")

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    dataset = Dataset(data=[{"image": sample_file}], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)

    batch = next(iter(dataloader))
    image = batch["image"]  # shape [1, 1, D, H, W]
    print("Loaded Image Shape:", image.shape)

    slice_index = image.shape[2] // 2
    image_slice = image[0, 0, slice_index].cpu().numpy()
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"NLST sample slice {slice_index}")
    plt.axis("off")
    plt.show()

    print("NLST-files are being used. For Model-Zoo Inferenz start with --run-bundle.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NLST + MONAI Model-Zoo Detection Beginner Script")
    parser.add_argument(
        "--run-bundle",
        action="store_true",
        help="Start the MONAI lung_nodule_ct_detection Bundle on NLST/imagesTr.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=1,
        help="Number of NLST volumes for the bundle inference. Test with a small number first.",
    )
    parser.add_argument(
        "--start-case",
        type=int,
        default=0,
        help="Start index for NLST cases to process. Useful for batch processing.‚",
    )
    args = parser.parse_args()

    if args.run_bundle:
        run_bundle_inference(args.max_cases, args.start_case)
    else:
        preview_first_volume()


if __name__ == "__main__":
    main()
