# MobileNetV2 Food Classifier -- Technical Documentation

## Overview

Image classification model for food content moderation. Uses a **MobileNetV2-0.35** backbone (ImageNet-pretrained) with frozen-backbone transfer learning followed by a fine-tuning stage.

| | |
|---|---|
| **Framework** | TensorFlow / Keras |
| **Backbone** | MobileNetV2 alpha=0.35 (ImageNet-pretrained) |
| **Head** | GlobalAveragePooling2D → BatchNorm → Dropout(0.3) → Dense(N, softmax, L2=1e-4) |
| **Input** | RGB image, 224×224 |
| **Output** | Softmax over the food classes detected at training time (9 typical: 8 food + `Other`) |
| **Mobile format** | TFLite float / fp16 / int8 + TFJS |
| **HF model repo** | [`maia2000/mobilenetv2-food`](https://huggingface.co/maia2000/mobilenetv2-food) |
| **HF dataset repo** | [`maia2000/food-classifier-dataset`](https://huggingface.co/datasets/maia2000/food-classifier-dataset) |

---

## 1. Why MobileNetV2-0.35

- ~410 k backbone parameters — fits under 1 MB as a TFLite fp16/int8 export
- Native TFLite + TFJS conversion, usable on mobile and in browser
- ImageNet features transfer cleanly to food imagery
- Explicit `Other` (non-food) class acts as a reject bin, which stops the model from giving high-confidence food predictions on random inputs

Compared to the ViT model in this repo: pure CNN, single-image, much smaller, much faster inference.

---

## 2. Classes

The number of classes is detected at training time from folder layout in the dataset directory. A typical setup after cleanup has 9 classes: 8 food labels plus `Other` (downloaded from Caltech-101 via `tools/download_other_class.py`).

Class merging is applied by the split tooling — e.g. `Donuts → Donut` via `CLASS_MERGE_MAP` in `tools/split_dataset.py`.

---

## 3. Training pipeline

The two entry points:

- **Local / server**: `python main.py --action train` (two-stage pipeline defined in `train/train.py`).
- **Colab**: `mobilenet_v2_colab.ipynb` — generated from `scripts/build_colab_notebooks.py`, disconnect-resilient with Drive checkpoints.

### Stages

1. **Stage 1** — backbone frozen, head only. Stabilises the head on the dataset.
2. **Stage 2** — fine-tune (backbone unfrozen, low LR). Optional, controlled by `train_config.fine_tune`.

### Hyperparameters (defaults in `config.yaml`)

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Stage 1 LR | 1e-3 |
| Stage 2 LR | 1e-5 |
| Loss | `sparse_categorical_crossentropy` |
| Epochs | 30 stage 1 + 30 stage 2 (budget), early-stops on `val_loss` patience 3 |
| Augmentation | RandomFlip(h) + RandomRotation(0.15) + RandomZoom(0.2) + RandomContrast(0.2) + RandomBrightness(0.2) + RandomTranslation(0.1) |
| Preprocessing | `mobilenet_v2.preprocess_input` (maps pixels to [-1, 1]) |
| Split | 70 / 15 / 15 (Train / Val / Test), fixed `seed=42`, SHA-256 leak check in `tools/split_dataset.py` |

### Dataset preparation

`rebuild_clean_split_before_train: true` in `config.yaml` triggers `tools/split_dataset.build_clean_split` before training. Input: `raw_dataset_dir` (class-folder layout). Output: `dataset_dir` with `Train/Val/Test/<class>/*`. Cross-class duplicates (same SHA-256 in multiple classes) are rejected upstream by `tools/clean_cross_class_duplicates.py`.

---

## 4. Mobile export

`tools/convert_model.py` produces, under `exports/`:

- `model.keras` — full Keras model (inference-only, augmentation layers stripped)
- `tflite/model.tflite` — float TFLite
- `tflite/model_fp16.tflite` — fp16 TFLite (recommended for deployment)
- `tflite/model_int8.tflite` — int8-quantized TFLite (smallest)
- `tfjs/` — TFJS graph model (for browser)
- `config.json`, `labels.json` — metadata consumed by the demo app

`tools/validate_exports.py` then runs all three TFLite formats plus Keras against the test split and writes `validation_metrics.json` + HTML/PDF reports.

---

## 5. App integration (`app_mockup_demo/app.py`)

`predict_mobilenetv2` loads `model.tflite` from `app_mockup_demo/models/` (downloadable via the HF panel in the sidebar). Standard softmax argmax over the class list shipped in `labels.json`.

---

## 6. Files

| Path | Purpose |
|---|---|
| `mobilenet_v2_colab.ipynb` | Generated Colab training notebook (do not hand-edit; regenerate via `scripts/build_colab_notebooks.py`) |
| `main.py` | Action dispatcher (`train / eval / validation / eval_tflite / test`) |
| `train/train.py` | Two-stage training implementation |
| `convert_to_tflite.py` | Standalone TFLite converter |
| `tools/convert_model.py` | All-in-one export pipeline (Keras + TFLite + TFJS + config.json) |
| `tools/validate_exports.py` | Cross-format accuracy + throughput validation |
| `tools/download_other_class.py` | Downloads Caltech-101 non-food images for the `Other` class |
| `tools/split_dataset.py` | Hash-based stratified split with leak detection |
| `tools/clean_cross_class_duplicates.py` | Pre-split dedup across classes |
| `config.yaml` | Hyperparameter + dataset paths |
| `requirements.txt` | Full training deps |
| `requirements-fetch-only.txt` | Slim deps for the dataset crawler in `tools/` |

See [`DATASET.md`](../../DATASET.md) for the dataset structure and class layout.
