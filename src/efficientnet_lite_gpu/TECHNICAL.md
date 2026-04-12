# EfficientNet-B0 Food Classifier -- Technical Documentation

## Overview

Image classification model for food content moderation. Classifies single images into **16 categories** across 3 health groups using an EfficientNet-B0 backbone with two-stage transfer learning.

**Framework:** TensorFlow / Keras
**Task:** 16-class classification (8 healthy + 7 unhealthy + 1 not_food)
**Input:** Single image (224x224)
**Output:** Class probabilities (softmax over 16 classes), rollup to healthy/unhealthy/not_food

---

## 1. Why EfficientNet-B0

EfficientNet uses compound scaling (depth, width, resolution scaled together) to achieve high accuracy with fewer parameters than comparable architectures. We chose B0 specifically because:

- **Size:** ~4M parameters -- the smallest EfficientNet variant, suitable for mobile deployment
- **Accuracy:** Achieves 77.1% top-1 on ImageNet -- better than ResNet-50 (76.1%) with 7x fewer parameters
- **Efficiency:** Designed via neural architecture search (NAS) to maximize accuracy per FLOP
- **Transfer learning:** Strong ImageNet pretrained weights transfer well to food classification
- **TFLite support:** Native TensorFlow model converts cleanly to TFLite with quantization for Android/iOS
- **Two-stage training:** Frozen backbone + head training, then full fine-tuning, prevents catastrophic forgetting

Compared to the ViT model in this project, EfficientNet is a pure CNN -- no temporal modeling. It classifies single images, not video sequences. This makes it simpler, faster at inference, and better suited when only a single frame is available.


---

## 2. Classes

### 2.1 Fine-Grained Classes (16)

Identical to the ViT model so both can be compared on the same dataset:

| # | Class | Health Group | Description |
|---|---|---|---|
| 1 | `fruits` | healthy | Whole fruits, fruit bowls, sliced fruit |
| 2 | `vegetables` | healthy | Cooked and raw vegetables |
| 3 | `salads` | healthy | Green salads, grain salads, mixed salads |
| 4 | `seafood` | healthy | Grilled/baked fish, sushi, shrimp, shellfish |
| 5 | `grilled_meat` | healthy | Lean grilled meats, chicken breast, steak |
| 6 | `grain_bowls` | healthy | Quinoa, rice bowls, oatmeal, buddha bowls |
| 7 | `soups` | healthy | Vegetable soups, miso, broths, stews |
| 8 | `smoothies` | healthy | Smoothies, fresh juices, healthy drinks |
| 9 | `burgers` | unhealthy | Cheeseburgers, hamburgers, hot dogs |
| 10 | `pizza` | unhealthy | All pizza types, calzones |
| 11 | `fried_food` | unhealthy | Fried chicken, fries, onion rings, nuggets |
| 12 | `desserts` | unhealthy | Cakes, donuts, pastries, pies, cookies |
| 13 | `candy_sweets` | unhealthy | Candy, chocolate, ice cream, frozen treats |
| 14 | `salty_snacks` | unhealthy | Chips, nachos, pretzels, cheese puffs |
| 15 | `sugary_drinks` | unhealthy | Soda, milkshakes, energy drinks, frappuccinos |
| 16 | `not_food` | not_food | People, animals, objects, nature, food-adjacent items |

### 2.2 Health-Label Rollup

```python
HEALTH_LABELS = {
    "fruits": "healthy", "vegetables": "healthy", "salads": "healthy",
    "seafood": "healthy", "grilled_meat": "healthy", "grain_bowls": "healthy",
    "soups": "healthy", "smoothies": "healthy",
    "burgers": "unhealthy", "pizza": "unhealthy", "fried_food": "unhealthy",
    "desserts": "unhealthy", "candy_sweets": "unhealthy",
    "salty_snacks": "unhealthy", "sugary_drinks": "unhealthy",
    "not_food": "not_food",
}
```

---

## 3. Model Architecture

### 3.1 High-Level Pipeline

```
Image (224x224x3)
  |
  v
[Data Augmentation] -- RandomFlip, Rotation, Zoom, Contrast (training only)
  |
  v
[EfficientNet-B0 Backbone] -- pretrained ImageNet, frozen in Stage 1
  |  Input:  (B, 224, 224, 3)
  |  Output: (B, 7, 7, 1280)
  v
[GlobalAveragePooling2D]
  |  Output: (B, 1280)
  v
[BatchNormalization]
  |
  v
[Dropout(0.2)]
  |
  v
[Dense(16, softmax)]
  |
  v
Output: (B, 16) probabilities
```

### 3.2 Two-Stage Transfer Learning

**Stage 1 -- Head training (backbone frozen):**
- Only the classification head (GAP + BN + Dropout + Dense) is trained
- Backbone weights are frozen to preserve ImageNet features
- Higher learning rate (1e-2) since head is randomly initialized
- EarlyStopping patience=8, ReduceLROnPlateau patience=3

**Stage 2 -- Full fine-tuning:**
- Entire model is unfrozen except BatchNormalization layers (kept frozen to preserve running statistics)
- Much lower learning rate (1e-4, i.e. lr/100) to avoid destroying pretrained features
- EarlyStopping patience=5, shorter since the model is already close to convergence

This two-stage approach is critical because:
- Training the full model from scratch with a random head would send noisy gradients into the backbone
- The frozen head-only stage lets the classifier head learn a reasonable mapping first
- Fine-tuning then adjusts the backbone features specifically for food classification

### 3.3 Backbone Variants Supported

| Model | Params | Top-1 ImageNet | Config key |
|---|---|---|---|
| EfficientNet-B0 | ~4M | 77.1% | `efficientnet-b0` |
| EfficientNet-B1 | ~6.5M | 79.1% | `efficientnet-b1` |
| EfficientNet-B2 | ~7.7M | 80.1% | `efficientnet-b2` |
| EfficientNet-B3 | ~10.7M | 81.6% | `efficientnet-b3` |

B0 is the default. Higher variants trade size for accuracy.

---

## 4. Dataset

### 4.1 Data Collection

~600 images per class across 16 classes (~9,600 total), downloaded via DuckDuckGo image search using the `tools/fetch_google_dataset.py` script.

- **27 keywords per class** (51 for not_food) with diverse search terms
- **~22 images per keyword** (target 600/class, balanced across classes)
- **DuckDuckGo** as default search engine (also supports Bing, Google via icrawler)
- **Balancing:** After download, all classes are trimmed to the same count (the minimum)

### 4.2 Data Source: HuggingFace

The dataset is hosted on HuggingFace at `maia2000/efficientnet-food-dataset` so teammates can download it directly. The notebook defaults to HuggingFace download (`USE_HF_DATASET = True`).

### 4.3 Dataset Format

```
train/dataset/
  Train/
    fruits/          # ~600 images
    vegetables/      # ~600 images
    ...
    not_food/        # ~600 images
  Test/
    fruits/
    vegetables/
    ...
    not_food/
```

Unlike the ViT model (which uses video-level frame naming), EfficientNet uses flat image folders. Classes are auto-discovered from directory names by `tf.keras.utils.image_dataset_from_directory()`.

### 4.4 Data Augmentation (training only)

| Augmentation | Parameters |
|---|---|
| `RandomFlip` | horizontal |
| `RandomRotation` | 0.05 |
| `RandomZoom` | 0.1 |
| `RandomContrast` | 0.1 |

Augmentation is applied as a Keras layer inside the model, so it runs on GPU and is automatically disabled at inference time.

### 4.5 Data Splitting

| Split | Ratio | Method |
|---|---|---|
| Train | 80% | `image_dataset_from_directory(validation_split=0.2, subset="training")` |
| Validation | 20% | `image_dataset_from_directory(validation_split=0.2, subset="validation")` |
| Test | separate folder | Independent `Test/` directory |

---

## 5. Training

| Parameter | Value | Why |
|---|---|---|
| Optimizer | Adam | Simple, effective default for transfer learning |
| Stage 1 LR | 1e-2 | High LR for randomly initialized head |
| Stage 2 LR | 1e-4 | Low LR to fine-tune without destroying pretrained weights |
| Loss | sparse_categorical_crossentropy | Integer labels, standard multi-class loss |
| Stage 1 epochs | 50 (max) | EarlyStopping usually triggers around 15-20 |
| Stage 2 epochs | 50 (max) | EarlyStopping usually triggers around 10-15 |
| Batch size | 8 | Fits Colab GPU memory |
| EarlyStopping | monitor=val_loss, patience=8 (S1) / 5 (S2) | Prevents overfitting |
| ReduceLROnPlateau | factor=0.5, patience=3, min_lr=1e-6 | Adapts LR when validation plateaus |
| Dropout | 0.2 | Light regularization before classifier |
| BatchNorm frozen in S2 | Yes | Preserves ImageNet running statistics |

### 5.1 Google Drive Checkpoints

When training on Google Colab, the trained model is synced to Google Drive (`/content/drive/MyDrive/whispr-checkpoints/efficientnet/`) after training completes. If the Colab runtime disconnects, the notebook restores the model from Drive on next run.

---

## 6. Evaluation

- Precision, Recall, F1 per class -- **do not rely on accuracy alone**
- Confusion matrix (16x16)
- Health-level rollup: precision/recall/F1 and confusion matrix at healthy/unhealthy/not_food level
- Per-class F1 bar chart
- Overall performance metrics bar chart

All evaluation artifacts are saved to `train/results/evaluation_results/`.

---

## 7. Export & Mobile Deployment

### 7.1 Export Formats

| Format | File | Size (approx.) | Target |
|---|---|---|---|
| Keras | `.h5` | ~17 MB | Server / GPU inference (includes augmentation layer) |
| TFLite | `.tflite` | ~4 MB | Android / iOS mobile (augmentation stripped) |
| TFLite float16 | `.tflite` | ~8 MB | Mobile with float16 precision |
| TFLite quantized | `.tflite` | ~4 MB | Mobile with dynamic range quantization |

### 7.2 Conversion Pipeline

The training model includes a `data_augmentation` layer that must be stripped before mobile deployment (augmentation is only used during training). The conversion flow:

```
BestModelEfficientNetLite.h5  (Keras, ~17 MB, includes data_augmentation)
  |
  v
[Strip data_augmentation layer]  -- rebuild inference-only graph
  |
  v
[Export to SavedModel]  -- TensorFlow SavedModel format
  |
  v
[TFLiteConverter]  -- apply optimizations
  |  Options:
  |    --quantize     dynamic range quantization (smallest, ~4 MB)
  |    (default)      float32 (largest, ~16 MB)
  v
BestModelEfficientNetLite_inference.tflite
```

### 7.3 How to Convert

```bash
# Default (no quantization)
python convert_to_tflite.py --model BestModelEfficientNetLite.h5

# With dynamic range quantization (recommended for mobile)
python convert_to_tflite.py --model BestModelEfficientNetLite.h5 --quantize

# Custom output path
python convert_to_tflite.py --model BestModelEfficientNetLite.h5 --output mobile_model.tflite --quantize
```

### 7.4 Mobile Inference

On mobile, the TFLite model expects:
- **Input:** `float32` tensor of shape `(1, 224, 224, 3)` -- RGB image, pixel values in `[0, 255]`
- **Output:** `float32` tensor of shape `(1, 16)` -- softmax probabilities for each class
- **Preprocessing:** Resize to 224x224, convert to float32, no normalization needed (EfficientNet has built-in preprocessing)

Example Android/iOS inference:

```
1. Load image from camera/gallery
2. Resize to 224x224
3. Convert to float32 array [1, 224, 224, 3]
4. Run TFLite interpreter
5. Read output [1, 16] probabilities
6. Map argmax to class name via labels.json
7. Map class name to health group via HEALTH_LABELS
```

### 7.5 Model Card

The exported model on HuggingFace (`zeyuai/efficientnet-food-classifier`) includes:
- `BestModelEfficientNetLite.keras` -- full Keras model
- `tflite/model.tflite` -- quantized TFLite model
- `tflite/model_float16.tflite` -- float16 TFLite model
- `labels.json` -- class names and id mappings
- `config.json` -- model metadata (input size, preprocessing, classes)
- `validation_metrics.json` -- test set performance numbers

---

## 8. Project Structure

```
src/efficientnet_lite_gpu/
  train/
    train.py                    # Full training pipeline (2-stage transfer learning)
  validation/
    validation.py               # Keras model inference + health-label predictions
    validation_tflite.py        # TFLite model inference
  test/
    test.py                     # Dataset image validation (corrupt file scanner)
  tools/
    configuration_generator.py  # Generates config.yaml from CLI args
    config_validator.py         # Validates config.yaml schema
    hardware_test.py            # GPU/CUDA/dependency checks
    tools_nvidia_cuda.py        # NVIDIA driver/nvcc verification
    fetch_google_dataset.py     # Image dataset downloader (DuckDuckGo/Bing/Google)
    dataset_download_config.yaml # 16-class search keywords
  main.py                       # CLI entry point (train | eval | eval_tflite | test)
  convert_to_tflite.py          # Keras -> TFLite conversion with quantization
  config.yaml                   # Training configuration
  efficientnet_colab.ipynb      # Google Colab notebook
  requirements.txt              # Python dependencies
  TECHNICAL.md                  # This file
```

---

## 9. Usage

```bash
# Full pipeline via CLI
python main.py --action train --model efficientnet-b0

# Evaluation
python main.py --action eval --model efficientnet-b0

# TFLite validation
python main.py --action eval_tflite --model efficientnet-b0

# Dataset image validation (scan for corrupt files)
python main.py --action test --model efficientnet-b0

# Download dataset
python -m tools.fetch_google_dataset
python -m tools.fetch_google_dataset --dry-run  # preview only

# Convert to TFLite
python convert_to_tflite.py --model BestModelEfficientNetLite.h5 --quantize

# Google Colab
# Open efficientnet_colab.ipynb -- handles everything automatically.
```

---

## 10. EfficientNet vs ViT -- Comparison

Both models are trained on the same 16 classes with the same health-label rollup so they can be directly compared.

| Aspect | EfficientNet-B0 | MobileViT-XXS + BiLSTM |
|---|---|---|
| Framework | TensorFlow / Keras | PyTorch |
| Input | Single image | Video (8 frames) |
| Params | ~4M | ~1.3M + LSTM |
| Architecture | Pure CNN (compound-scaled) | Hybrid CNN + Transformer + LSTM |
| Temporal modeling | None | BiLSTM over frame features |
| Training strategy | 2-stage transfer learning | End-to-end with LR warmup + cosine |
| Mobile export | TFLite (native) | TorchScript, ONNX, CoreML, TFLite |
| Best for | Single image classification | Video stream moderation |

---

## 11. HuggingFace Repos

| Repo | Type | Content |
|---|---|---|
| `maia2000/efficientnet-food-dataset` | dataset | 16-class images (`Train/<class>/*.jpg`, `Test/<class>/*.jpg`) |
| `zeyuai/efficientnet-food-classifier` | model | Trained .h5 + .tflite exports |
