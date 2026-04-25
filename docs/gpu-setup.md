# Setup GPU

## NVIDIA CUDA

Les outils GPU sont dans `src/efficientnet_lite_gpu/tools/`.

```bash
python tools/hardware_test.py    # Tester le GPU
python tools/config_validator.py # Valider la config
```

## Sans GPU

Les modèles fonctionnent aussi en CPU, mais l'entraînement sera plus lent.
