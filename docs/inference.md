# Inférence

## Image

```bash
cd src/mobilenet_v2_small
python main.py --action eval --image path/to/image.jpg
```

## Vidéo (ViT)

```bash
cd src/vit_video
python inference.py --model models/best_food_classifier.pth --video clip.mp4
```
