# Comparaison des modèles

## Performance

| Modèle | Taille | Latence | Précision | Plateforme |
|--------|--------|---------|-----------|------------|
| MobileNetV2 | < 1 MB | < 5ms | ~90% | Edge/Mobile |
| MobileNetV3 | < 10 MB | < 10ms | ~92% | Mobile |
| ViT-Video | ~85 MB | ~50ms/frame | ~88% | Serveur |

## Quand utiliser lequel

```
Image fixe ──▶ MobileNetV2 (rapide) ou MobileNetV3 (précis)
Vidéo ──▶ ViT-Video (multi-frames)
```
