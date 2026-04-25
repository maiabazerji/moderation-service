# Export TFLite

## Flux d'export

```
Modèle Keras ──▶ convert_to_tflite.py ──▶ Modèle .tflite
                                                │
                                          Quantization
                                          (optionnel)
                                                │
                                          Déploiement mobile
```

## Commande

```bash
python convert_to_tflite.py --quantize
```
