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
ORIGINAL_DATA_DIR = os.path.join(PROJECT_ROOT, "NLST", "imagesTr")
CORRUPTED_DATA_DIR = os.path.join(PROJECT_ROOT, "NLST", "corrupted_imagesTr")
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "model-zoo", "models", "lung_nodule_ct_detection")
CHECKPOINT = os.path.join(BUNDLE_DIR, "models", "model.pt")
CPU_CHECKPOINT = os.path.join(BUNDLE_DIR, "models", "model_cpu.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "nlst_detection_outputs")


def require_module(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency '{module_name}' in {sys.executable}.\n"
            f"Use the project environment first, for example:\n"
            f"  conda activate monai-nlst\n"
            f"or run explicitly with:\n"
            f"  /Users/kathi/miniforge3/envs/monai-nlst/bin/python {os.path.basename(__file__)} ...\n"
            f"Install hint: {install_hint}"
        ) from exc


def ensure_cpu_checkpoint_if_needed() -> None:
    require_module("torch", "pip install torch")
    import torch

    if torch.cuda.is_available():
        return

    if os.path.exists(CPU_CHECKPOINT):
        return

    print("CUDA is not available. Creating CPU-compatible checkpoint model_cpu.pt ...")
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"), weights_only=True)
    torch.save(checkpoint, CPU_CHECKPOINT)


def default_output_filename(data_mode: str) -> str:
    if data_mode == "clean":
        return "nlst_lung_nodule_predictions.json"
    if data_mode == "corrupted":
        return "nlst_lung_nodule_corrupted_predictions.json"
    return "nlst_lung_nodule_predictions_custom.json"


def run_bundle_inference(
    max_cases: int,
    start_case: int,
    dataset_dir: str,
    output_filename: str,
) -> None:
    if not os.path.exists(BUNDLE_DIR):
        raise FileNotFoundError(f"MONAI Bundle not found: {BUNDLE_DIR}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Pretrained Checkpoint missing: {CHECKPOINT}")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    require_module("monai", "pip install 'monai[all]'")
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
        "--dataset_dir",
        dataset_dir,
        "--output_dir",
        OUTPUT_DIR,
        "--output_filename",
        output_filename,
    ]

    print("Starting MONAI Model-Zoo Inference:")
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {os.path.join(OUTPUT_DIR, output_filename)}")
    print(" ".join(command))
    subprocess.run(command, cwd=BUNDLE_DIR, check=True)


def preview_first_volume(dataset_dir: str) -> None:
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

    if not os.path.exists(dataset_dir):
        print("Please download the NLST dataset and extract it to NLST/imagesTr.")
        print("The folder should contain e.g., NLST/imagesTr/NLST_0001_0000.nii.gz.")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.nii.gz")))
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
        default=30,
        help="Number of NLST volumes for the bundle inference. Test with a small number first.",
    )
    parser.add_argument(
        "--start-case",
        type=int,
        default=0,
        help="Start index for NLST cases to process. Useful for batch processing.‚",
    )
    parser.add_argument(
        "--data-mode",
        choices=["clean", "corrupted"],
        default="clean",
        help="Use original NLST/imagesTr or generated NLST/corrupted_imagesTr.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional explicit dataset folder. Overrides --data-mode.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Output JSON filename inside nlst_detection_outputs.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = ORIGINAL_DATA_DIR if args.data_mode == "clean" else CORRUPTED_DATA_DIR
    output_filename = args.output_filename or default_output_filename(args.data_mode)

    if args.run_bundle:
        run_bundle_inference(args.max_cases, args.start_case, dataset_dir, output_filename)
    else:
        preview_first_volume(dataset_dir)


if __name__ == "__main__":
    main()
