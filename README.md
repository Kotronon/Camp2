# CAMP2

## MONAI Lung Nodule CT Detection Bundle

The Lung Nodule CT Detection MONAI bundle is included under this project so it can be opened from VSCode. Large model weight files are kept locally in the bundle folder but are ignored by Git, so they are not committed directly to GitHub.

Bundle path in this project:

```bash
./bundles/lung_nodule_ct_detection
```

Local model weights:

```bash
./bundles/lung_nodule_ct_detection/models/model.pt
./bundles/lung_nodule_ct_detection/models/model.ts
```

Environment used:

```bash
mkdir -p ~/monai_local
uv venv --python 3.10 ~/monai_local/monai_lung
source ~/monai_local/monai_lung/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install "monai[fire]" requests huggingface_hub
```

Download or refresh the bundle from the project root:

```bash
cd /Users/wangtaiding/Documents/CAMP2
python -m monai.bundle download --name lung_nodule_ct_detection --bundle_dir ./bundles
```

Verification:

```bash
ls ./bundles/lung_nodule_ct_detection
ls ./bundles/lung_nodule_ct_detection/models
```

Expected local model file:

```bash
model.pt
```
