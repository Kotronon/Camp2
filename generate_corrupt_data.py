'''
Generate corrupt data for testing purposes.
For that purpose, we will take the first 10 cases of the NLST dataset and corrupt them by setting the upper lung pixel values to the same value as air.
This will create a scenario where the upper lung is not visible, which can be used to test the robustness of our models against corrupted data.
At the same time, we will keep the lower lung intact, so that the models can still make predictions based on the lower lung, which is a common scenario 
in real-world data where some parts of the image may be corrupted while others are still usable.
As bonus we could cut some slices from the upper lung to create a scenario where the upper lung is partially visible, which can also be used to test the robustness of our models against corrupted data.
Or where we cut the lower part of the lung, which can also be used to test the robustness of our models against corrupted data.

Document per image: filename; filename, corruption_type, z_start, z_end, removed_slices, removed_fraction

Important: Remove slides from lungs (use NLST/masksTR to identify lung area) and not from the upper part of the image, which may contain other structures. 
Otherwise, we would create a scenario that is not realistic and may not be useful for testing the robustness of our models against corrupted data.
'''

import os
import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "NLST", "imagesTr")
CORRUPTED_DATA_DIR = os.path.join(PROJECT_ROOT, "NLST", "corrupted_imagesTr")
os.makedirs(CORRUPTED_DATA_DIR, exist_ok=True)

def corrupt_upper_part_nlst_data(num_cases: int = 10, start_case: int = 0, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        # get upper part of the lung and set it to air (-1000 HU)
        # This would require loading the lung mask and identifying the upper lung region
        lung_mask_file = case_file.replace("imagesTr", "masksTr")
        if not os.path.exists(lung_mask_file):
            print(f"Warning: Lung mask not found for {case_file}. Skipping corruption for this case.")
            continue
        lung_mask = nib.load(lung_mask_file).get_fdata()
        # Identify the upper lung region using the lung mask
        # For simplicity, we will assume the upper lung is in the upper half of the image
        upper_lung_region = lung_mask > 0
        if not np.any(upper_lung_region):
            print(f"Warning: No upper lung region found for {case_file}. Skipping corruption for this case.")
            continue

        # Set upper lung pixel values to air (e.g., -1000 HU) only in lung regions
        upper_lung_threshold = data.shape[2] // 2
        # Create mask for upper lung region
        upper_region_mask = np.zeros_like(data, dtype=bool)
        upper_region_mask[:, :, :upper_lung_threshold] = True
        # Apply corruption only where lung_mask > 0 and in upper region
        data[(lung_mask > 0) & upper_region_mask] = -1000
        
        # Calculate modified slices (z-slices with at least one modified voxel)
        modified_slices_mask = np.any((lung_mask > 0) & upper_region_mask, axis=(0, 1))
        num_modified_slices = np.sum(modified_slices_mask)
        removed_fraction = num_modified_slices / upper_lung_threshold
        
        # Save the corrupted image
        corrupted_img = nib.Nifti1Image(data, img.affine, img.header)
        corrupted_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file))
        nib.save(corrupted_img, corrupted_case_file)
        print(f"Corrupted {case_file} and saved to {corrupted_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "upper_lung_corruption",
            "z_start": 0,
            "z_end": upper_lung_threshold,
            "removed_slices": num_modified_slices,
            "removed_fraction": removed_fraction
        }
        
def corrupt_partial_upper_part_nlst_data(num_cases: int = 10, start_case: int = 0, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        
        # get upper part of the lung and set it to air (-1000 HU)
        # This would require loading the lung mask and identifying the upper lung region
        lung_mask_file = case_file.replace("imagesTr", "masksTr")
        if not os.path.exists(lung_mask_file):
            print(f"Warning: Lung mask not found for {case_file}. Skipping corruption for this case.")
            continue
        lung_mask = nib.load(lung_mask_file).get_fdata()
        # Identify the upper lung region using the lung mask
        # For simplicity, we will assume the upper lung is in the upper half of the image
        upper_lung_region = lung_mask > 0
        if not np.any(upper_lung_region):
            print(f"Warning: No upper lung region found for {case_file}. Skipping corruption for this case.")
            continue
        
        # Set upper lung pixel values to air (e.g., -1000 HU) for the upper quarter of the image
        upper_lung_threshold = data.shape[2] // 4
        upper_region_mask = np.zeros_like(data, dtype=bool)
        upper_region_mask[:, :, :upper_lung_threshold] = True
        data[(lung_mask > 0) & upper_region_mask] = -1000
        
        # Calculate modified slices (z-slices with at least one modified voxel)
        modified_slices_mask = np.any((lung_mask > 0) & upper_region_mask, axis=(0, 1))
        num_modified_slices = np.sum(modified_slices_mask)
        removed_fraction = num_modified_slices / upper_lung_threshold
        
        # Save the corrupted image
        corrupted_img = nib.Nifti1Image(data, img.affine, img.header)
        corrupted_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file))
        nib.save(corrupted_img, corrupted_case_file)
        print(f"Partially corrupted {case_file} and saved to {corrupted_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "upper_lung_corruption",
            "z_start": 0,
            "z_end": upper_lung_threshold,
            "removed_slices": num_modified_slices,
            "removed_fraction": removed_fraction
        }

def corrupt_lower_part_nlst_data(num_cases: int = 10, start_case: int = 0, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        # get upper part of the lung and set it to air (-1000 HU)
        # This would require loading the lung mask and identifying the upper lung region
        lung_mask_file = case_file.replace("imagesTr", "masksTr")
        if not os.path.exists(lung_mask_file):
            print(f"Warning: Lung mask not found for {case_file}. Skipping corruption for this case.")
            continue
        lung_mask = nib.load(lung_mask_file).get_fdata()
        # Identify the upper lung region using the lung mask
        # For simplicity, we will assume the upper lung is in the upper half of the image
        upper_lung_region = lung_mask > 0
        if not np.any(upper_lung_region):
            print(f"Warning: No upper lung region found for {case_file}. Skipping corruption for this case.")
            continue
        # Set lower lung pixel values to air (e.g., -1000 HU) for the lower quarter of the image
        lower_lung_threshold = data.shape[2] * 3 // 4
        lower_region_mask = np.zeros_like(data, dtype=bool)
        lower_region_mask[:, :, lower_lung_threshold:] = True
        data[(lung_mask > 0) & lower_region_mask] = -1000
        
        # Calculate modified slices (z-slices with at least one modified voxel)
        modified_slices_mask = np.any((lung_mask > 0) & lower_region_mask, axis=(0, 1))
        num_modified_slices = np.sum(modified_slices_mask)
        total_slices_in_region = data.shape[2] - lower_lung_threshold
        removed_fraction = num_modified_slices / total_slices_in_region
        
        # Save the corrupted image
        corrupted_img = nib.Nifti1Image(data, img.affine, img.header)
        corrupted_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file))
        nib.save(corrupted_img, corrupted_case_file)
        print(f"lower corrupted {case_file} and saved to {corrupted_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "lower_lung_corruption",
            "z_start": lower_lung_threshold,
            "z_end": data.shape[2] - 1,
            "removed_slices": num_modified_slices,
            "removed_fraction": removed_fraction
        }
        
def crop_images_z_axis(num_cases: int = 10, start_case: int = 0, start: int = 0, percentage: float = 0.5, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        # crop the image to the upper half
        cropped_data = data[:, :, start:start + int(data.shape[2] * percentage)]
        cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        cropped_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file).replace(".nii.gz", "_cropped.nii.gz"))
        nib.save(cropped_img, cropped_case_file)
        print(f"Cropped {case_file} and saved to {cropped_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "cropping",
            "z_start": start,
            "z_end": start + int(data.shape[2] * percentage),
            "removed_slices": data.shape[2] - int(data.shape[2] * percentage),
            "removed_fraction": 1 - percentage
        }

def crop_images_x_axis(num_cases: int = 10, start_case: int = 0, start: int = 0, percentage: float = 0.5, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        # crop the image to the upper half
        cropped_data = data[start:start + int(data.shape[0] * percentage), :, :]
        cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        cropped_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file).replace(".nii.gz", "_cropped_x.nii.gz"))
        nib.save(cropped_img, cropped_case_file)
        print(f"Cropped {case_file} and saved to {cropped_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "cropping_x",
            "z_start": start,
            "z_end": start + int(data.shape[0] * percentage),
            "removed_slices": data.shape[0] - int(data.shape[0] * percentage),
            "removed_fraction": 1 - percentage
        }
        
def crop_images_y_axis(num_cases: int = 10, start_case: int = 0, start: int = 0, percentage: float = 0.5, documentation: pd.DataFrame = None) -> None:
    case_files = sorted(glob(os.path.join(DATA_DIR, "*.nii.gz")))[start_case:start_case + num_cases]
    for case_file in case_files:
        img = nib.load(case_file)
        data = img.get_fdata()
        # crop the image to the upper half
        cropped_data = data[:, start:start + int(data.shape[1] * percentage), :]
        cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        cropped_case_file = os.path.join(CORRUPTED_DATA_DIR, os.path.basename(case_file).replace(".nii.gz", "_cropped_y.nii.gz"))
        nib.save(cropped_img, cropped_case_file)
        print(f"Cropped {case_file} and saved to {cropped_case_file}")
        # documentation
        documentation.loc[len(documentation)] = {
            "filename": os.path.basename(case_file),
            "corruption_type": "cropping_y",
            "z_start": start,
            "z_end": start + int(data.shape[1] * percentage),
            "removed_slices": data.shape[1] - int(data.shape[1] * percentage),
            "removed_fraction": 1 - percentage
        }
        
if __name__ == "__main__":
    # documentation cv:
    documentation = pd.DataFrame(columns=["filename", "corruption_type", "z_start", "z_end", "removed_slices", "removed_fraction"])
    #corrupt_upper_part_nlst_data(num_cases=10, start_case=0, documentation=documentation)
    #corrupt_partial_upper_part_nlst_data(num_cases=10, start_case=10, documentation=documentation)
    #corrupt_lower_part_nlst_data(num_cases=10, start_case=20, documentation=documentation)
    crop_images_z_axis(num_cases=1, start_case=0, start=0, percentage=0.5, documentation=documentation)
    crop_images_x_axis(num_cases=1, start_case=0, start=0, percentage=0.5, documentation=documentation)
    crop_images_y_axis(num_cases=1, start_case=0, start=0, percentage=0.5, documentation=documentation)
    documentation.to_csv(os.path.join(os.path.join(PROJECT_ROOT, "nlst_detection_outputs"), "corruption_documentation.csv"), index=False)