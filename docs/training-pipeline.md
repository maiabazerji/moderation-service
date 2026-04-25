# Pipeline d'entraînement

## Étapes

```
Dataset HuggingFace
     │
     ▼
┌──────────┐
│ Download │
│ + Split  │
└────┬─────┘
     │
     ▼
┌──────────┐
│ Training │
│ (GPU)    │
└────┬─────┘
     │
     ▼
┌──────────┐
│ Eval     │
│ Metrics  │
└────┬─────┘
     │
     ▼
┌──────────┐
│ Export   │
│ TFLite   │
└──────────┘
```
