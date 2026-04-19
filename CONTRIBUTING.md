# Contribuer au Moderation Service

## Stack

- Python 3.10+
- TensorFlow / PyTorch
- FastAPI

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement d'un modèle

```bash
cd src/mobilenet_v2_small
python main.py --action train
```

## Conventions

- Branches : `WHISPR-XXX-description`
- Tests avant chaque PR

## GPU

Pour l'entraînement avec GPU, voir `src/efficientnet_lite_gpu/tools/` pour les outils NVIDIA CUDA.

## Modèles disponibles

| Modèle | Dossier | Usage |
|--------|---------|-------|
| MobileNetV2 | `src/mobilenet_v2_small/` | Classification image (edge) |
| MobileNetV3 | `src/mobilenet_v3_small/` | Classification image (mobile) |
| ViT-Video | `src/vit_video/` | Classification vidéo |
