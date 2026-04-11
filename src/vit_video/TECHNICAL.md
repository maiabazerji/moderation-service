# ViT Video Classifier -- Technical Documentation

## Overview

Fine-grained video/image classification model for food content moderation. Classifies into **16 categories** across 3 health groups using a MobileViT-XXS backbone with a bidirectional LSTM for temporal aggregation.

**Framework:** PyTorch
**Task:** 16-class classification (8 healthy + 7 unhealthy + 1 not_food)
**Input:** Video or sequence of frames
**Output:** Class probabilities (softmax over 16 classes), rollup to healthy/unhealthy/not_food

---

## 1. Classes

### 1.1 Fine-Grained Classes (16)

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

### 1.2 Health-Label Rollup

The 16 fine-grained predictions are mapped to 3 moderation groups for the API:

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

## 2. Model Architecture

### 2.1 High-Level Pipeline

```
Video (T frames)
  |
  v
[Frame Extraction] -- evenly-spaced sampling (default T=8)
  |
  v
[Per-Frame MobileViT-XXS Backbone] -- pretrained ImageNet feature extractor
  |  Input:  (B*T, 3, 224, 224)
  |  Output: (B*T, feat_dim)
  v
[Reshape] -- (B, T, feat_dim)
  |
  v
[Bidirectional LSTM] -- 1-layer, hidden_size = feat_dim/2 per direction
  |  Input:  (B, T, feat_dim)
  |  Output: (B, T, feat_dim)
  v
[Temporal Mean Pool] -- average across T timesteps
  |  Output: (B, feat_dim)
  v
[Dropout(0.4)]
  |
  v
[Linear Classifier] -- (feat_dim -> 16)
  |
  v
Output: (B, 16) logits
```

### 2.2 Backbone Options

| Backbone | Feature Dim | Source | When Selected |
|---|---|---|---|
| `mobilevit_xxs` | 320 | timm | CPU fallback (default) |
| `mobilevit_xs` | 384 | timm | Manual selection |
| `mobilevit_s` | 640 | timm | Manual selection |
| `vit_b_16` | 768 | torchvision / timm | GPU available (default) |
| `vit_b_32` | 768 | torchvision / timm | Manual selection |
| `vit_l_16` | 1024 | torchvision / timm | Manual selection |
| `vit_h_14` | 1280 | torchvision / timm | Manual selection |

### 2.3 Temporal Pooling Modes

| Mode | Operation | Description |
|---|---|---|
| `lstm` (default) | `BiLSTM(feat_dim, feat_dim//2, bidirectional=True) -> mean` | Learns temporal frame dependencies |
| `avg` | `feats.mean(dim=1)` | Simple mean pooling |
| `max` | `feats.max(dim=1)` | Max pooling |
| `conv1d` | `Conv1d(feat_dim, feat_dim, kernel=3) -> mean` | Learned temporal convolution |

### 2.4 Input / Output Shapes

| Tensor | Shape | Description |
|---|---|---|
| Input | `(B, 8, 3, 224, 224)` | Batch of videos: 8 frames, RGB, 224x224 |
| Backbone output | `(B*8, feat_dim)` | Per-frame features |
| LSTM output | `(B, 8, feat_dim)` | Contextualized frame features |
| After temporal pool | `(B, feat_dim)` | Video-level features |
| Output logits | `(B, 16)` | Raw class scores |

---

## 3. Dataset

### 3.1 Data Collection

~8,600 images across 16 classes, downloaded from Bing image search via `icrawler`. Each keyword produces ~20 images named with the `_frame_NNNN` convention (pseudo-videos).

- **~27 keywords per class** with diverse search terms
- **20 images per keyword** = ~540 images per class
- **not_food** includes hard negatives: empty plates, kitchen utensils, grocery aisles

### 3.2 Data Augmentation (training only)

| Augmentation | Parameters |
|---|---|
| `RandomResizedCrop` | scale=(0.6, 1.0), ratio=(0.8, 1.2) |
| `RandomHorizontalFlip` | p=0.5 |
| `RandomRotation` | degrees=15 |
| `RandomPerspective` | distortion_scale=0.2, p=0.3 |
| `ColorJitter` | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 |
| `GaussianBlur` | kernel_size=3, sigma=(0.1, 2.0) |
| `RandomErasing` | p=0.3, scale=(0.02, 0.2) |

**Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

### 3.3 Data Splitting

Video-level stratified splitting (no frame leakage):

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model training |
| Validation | 15% | Early stopping, LR scheduling |
| Test | 15% | Final evaluation |

---

## 4. Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-5, weight_decay=1e-3) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Class weighting | Inverse frequency per class |
| Gradient clipping | max_norm=1.0 |
| LR schedule | LinearLR warmup (3 epochs) + CosineAnnealingLR (eta_min=1e-7) |
| Early stopping | patience=7, min_delta=5e-5 |
| AMP | Enabled on CUDA |
| Epochs | 20 (notebook) / 25 (CLI default) |
| Batch size | 8 |
| Temporal pool | lstm |
| Dropout | 0.4 |

---

## 5. Evaluation

- Accuracy, Precision, Recall, F1 (macro-averaged) at fine-grained level
- Confusion matrix (16x16)
- Health-level rollup: accuracy and confusion matrix at healthy/unhealthy/not_food level
- K-fold cross-validation (video-grouped, stratified)
- Optional external data testing

---

## 6. Export Formats

| Format | File | Target |
|---|---|---|
| TorchScript | `.pt` | Mobile / embedded |
| ONNX | `.onnx` | Cross-platform (opset 17) |
| CoreML | `.mlpackage` | iOS 15+ |
| TFLite | `.tflite` | Android |

---

## 7. Project Structure

```
src/vit_video/
  models/vit.py              # MobileViTModel (backbone + BiLSTM + classifier)
  engine/trainer.py           # Training loop, AdamW, LR scheduler, AMP
  data/dataset.py             # VideoDataset, build_dataloaders
  data/splits.py              # Video-level splitting, manifest management
  utils/                      # Hardware, transforms, checkpoint utils
  train.py                    # Training entry point
  test.py                     # Evaluation entry point
  inference.py                # Video/webcam inference
  validate_model.py           # Leakage audit, k-fold CV, external test
  export_mobile.py            # Multi-format model export + model card
  upload_hf.py                # Hugging Face Hub upload
  run_pipeline.py             # End-to-end pipeline
  paths.py                    # Default directory constants
  _bootstrap.py               # sys.path setup for standalone scripts
  vit_video.ipynb             # Google Colab notebook
  requirements.txt            # Python dependencies
```

---

## 8. Usage

```bash
# Training
python train.py --dataset-dir food_data/frames --epochs 25 --temporal-pool lstm

# Evaluation
python test.py --model models/best_food_classifier.pth --dataset-dir food_data/frames

# Inference
python inference.py --video path/to/video.mp4 --model models/best_food_classifier.pth
python inference.py --webcam --model models/best_food_classifier.pth

# Export
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx

# Validation
python validate_model.py --dataset-dir food_data/frames --n-folds 5

# Google Colab
# Open vit_video.ipynb -- handles everything automatically.
```
