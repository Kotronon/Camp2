# NLST Lung Nodule Detection mit MONAI

Dieses Projekt implementiert die Detektion von Lungenknoten in CT-Scans basierend auf dem NLST-Dataset und dem MONAI Framework.

## Setup

1. **Umgebung aktivieren:**
   ```bash
   conda activate monai-nlst
   ```

2. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

   If `python nlst_detection_start.py --run-bundle ...` fails with missing `ignite`,
   `skimage`, or `einops`, install the updated requirements in the active environment:

   ```bash
   pip install -r requirements.txt
   ```

   In this workspace, the completed pipeline was run with `.venv/bin/python`
   because the `monai-nlst` Conda environment was missing these optional MONAI
   bundle dependencies.

3. **MONAI Model Zoo klonen:**
   ```bash
   git clone https://github.com/Project-MONAI/model-zoo.git
   ```

4. **Dataset herunterladen:**
   - Gehe zu: https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR
   - Lade das NLST-Dataset herunter und entpacke es in einen Ordner (z.B. `data/nlst/`).

## Nächste Schritte

1. **Dataset herunterladen und vorbereiten:**
   - Lade das NLST-Dataset herunter.
   - Erstelle einen Ordner `data/nlst/` und entpacke die Daten.
   - Aktualisiere `data_dir` im Skript.

2. **Modell laden:**
   - Das vorgegebene Modell hat Kompatibilitätsprobleme mit PyTorch/Torchvision-Versionen.
   - Versuche, die Versionen zu aktualisieren: `pip install torch==2.1.0 torchvision==0.16.0` (oder kompatible Versionen).
   - Oder verwende das Platzhalter-UNet und passe es für Detection an.

3. **Training und Evaluierung:**
   - Implementiere Training mit MONAI's Trainer.
   - Evaluiere Metriken wie mAP für Detection.

## Hinweise

- Das Skript verwendet aktuell ein einfaches UNet als Platzhalter.
- Für echte Detection brauchst du ein Modell wie RetinaNet oder Faster R-CNN, angepasst für 3D.
- Bei Problemen mit dem Bundle-Modell, kontaktiere MONAI-Support oder verwende ein alternatives Modell.

## Projektstruktur

- `nlst_detection_start.py`: Einstiegsskript für Modell-Loading und Inference.
- `requirements.txt`: Python-Abhängigkeiten.
- `model-zoo/`: Geklontes MONAI Model Zoo Repository.
- `data/`: (Erstelle diesen Ordner für das Dataset)

## Ressourcen

- [MONAI Dokumentation](https://docs.monai.io/)
- [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo)
- [NLST Dataset](https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR)

## Run

### Robustness pipeline for the final project

1. Generate 30 comparable corruptions:

   ```bash
   python generate_corrupt_data.py
   ```

   This writes:
   - `NLST/corrupted_imagesTr/*.nii.gz`
   - `nlst_detection_outputs/corruption_documentation.csv`

   Default corruption protocol:
   - 10 `lung_region_occlusion` cases: targeted lung-content ablation.
   - 10 `full_slice_dropout` cases: missing axial scan slab, including all anatomy.
   - 10 `low_dose_noise` cases: Gaussian HU noise in non-air/body voxels.

2. Run clean inference on the same first 30 cases:

   ```bash
   python nlst_detection_start.py --run-bundle --data-mode clean --max-cases 30 --start-case 0
   ```

3. Run corrupted inference on the generated cases:

   ```bash
   python nlst_detection_start.py --run-bundle --data-mode corrupted --max-cases 30 --start-case 0
   ```

   Re-run this step after changing the corruption generator. Existing corrupted prediction JSONs are not valid for newly generated corruptions.

4. Compare clean vs. corrupted predictions:

   ```bash
   python compare_nlst_predictions.py
   ```

   This writes:
   - `nlst_detection_outputs/original_vs_corrupted_summary.csv`
   - `nlst_detection_outputs/original_vs_corrupted_aggregates.csv`

5. Apply the simple post-processing mitigation:

   ```bash
   python mitigate_nlst_predictions.py
   python compare_nlst_predictions.py \
     --corrupted nlst_detection_outputs/nlst_lung_nodule_corrupted_predictions_mitigated.json \
     --output nlst_detection_outputs/original_vs_mitigated_summary.csv \
     --aggregate-output nlst_detection_outputs/original_vs_mitigated_aggregates.csv
   ```

6. Generate visual evidence for the report/slides:

   ```bash
   python visualize_failure_cases.py
   ```

   This writes PNGs to `nlst_detection_outputs/visual_evidence/`.

For this project, report detection robustness metrics instead of Dice: detection counts, disappeared detections, new detections, IoU matching, and score changes.
