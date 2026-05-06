# CAMP2

## MONAI Lung Nodule CT Detection Bundle

The Lung Nodule CT Detection MONAI bundle is downloaded locally instead of being committed to this repository, because the model files are large.

Local setup used:

```bash
mkdir -p ~/monai_local
cd ~/monai_local

uv venv --python 3.10 monai_lung
source monai_lung/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install "monai[fire]" requests huggingface_hub

python -m monai.bundle download --name lung_nodule_ct_detection --bundle_dir ./bundles
```

Downloaded bundle path:

```bash
~/monai_local/bundles/lung_nodule_ct_detection
```

Model weights:

```bash
~/monai_local/bundles/lung_nodule_ct_detection/models/model.pt
```

Verification:

```bash
ls ~/monai_local/bundles/lung_nodule_ct_detection
ls ~/monai_local/bundles/lung_nodule_ct_detection/models
```

Expected model file:

```bash
model.pt
```
