# Dataset

## Source

HuggingFace : `maia2000/food-classifier-dataset`

## Pipeline

```
HuggingFace ──▶ Download ──▶ Split train/val/test ──▶ Preprocessing
                                                           │
                                                     Data augmentation
                                                     (rotation, flip)
```

Voir `DATASET.md` à la racine pour les détails complets.
