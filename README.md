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

Run predictions by: 

python nlst_detection_start.py --run-bundle --max-cases 30 --start-case 0

To change the folder for the input-data change "dataset_dir" and "output_filename" accordingly.

For corrupting data see the script generate_corrupt_data.py. To change the amound of the different corrupted data, change the main accordingly