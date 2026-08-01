# Malaria Detection AI

A CNN that classifies microscopy images of blood cells as parasitized or uninfected with malaria (Plasmodium). ~94% accuracy.

## Model

An 8-conv-layer CNN (`model.py`) — 4 blocks of (Conv → BatchNorm → Conv → BatchNorm → MaxPool → Dropout), channels growing 32 → 64 → 128 → 256, followed by 2 fully connected layers down to a single sigmoid output. Takes 128×128 RGB cell images.

Trained in `Malaria_detection.ipynb`.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Or with Docker (`Dockerfile` + `Procfile` included for deployment):
```bash
docker build -t malaria-detection .
docker run -p 5000:5000 malaria-detection
```

## Dataset

NIH malaria cell image dataset — parasitized vs. uninfected blood smear images.

---
Learning project — not a substitute for clinical diagnosis.
