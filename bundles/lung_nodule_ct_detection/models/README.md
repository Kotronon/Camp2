# Model Weights

This directory is for local MONAI model weights downloaded with the Lung Nodule CT Detection bundle.

The large weight files are intentionally ignored by Git:

- model.pt
- model.ts

Download them locally with:

```bash
python -m monai.bundle download --name lung_nodule_ct_detection --bundle_dir ./bundles
```
