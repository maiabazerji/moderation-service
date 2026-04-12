# ViT Video Classifier -- Technical Documentation

## Overview

Fine-grained video/image classification model for food content moderation. Classifies into **16 categories** across 3 health groups using a MobileViT-XXS backbone with a bidirectional LSTM for temporal aggregation.

**Framework:** PyTorch
**Task:** 16-class classification (8 healthy + 7 unhealthy + 1 not_food)
**Input:** Video or sequence of frames
**Output:** Class probabilities (softmax over 16 classes), rollup to healthy/unhealthy/not_food

---

## 1. Why MobileViT-XXS

MobileViT combines the local feature extraction of MobileNetV2 with the global attention of Vision Transformers in a single lightweight architecture. We chose the XXS variant for several reasons:

- **Size:** ~1.3M parameters -- small enough for mobile/edge deployment while still accurate
- **Speed:** Designed for real-time inference on mobile CPUs (no GPU required on device)
- **Accuracy:** Outperforms pure CNNs of similar size on ImageNet thanks to self-attention
- **Temporal support:** Feature vectors are fixed-size (320-dim), making them easy to feed into a BiLSTM for video-level temporal aggregation -- something heavier ViT models would make impractical on-device
- **Pretrained on ImageNet:** Strong transfer learning baseline for food classification
- **Export-friendly:** Compatible with TorchScript, ONNX, CoreML, and TFLite

The BiLSTM on top allows the model to learn temporal dependencies between frames (e.g., a video panning across a plate of food), which a single-frame classifier would miss.

---

## 2. Classes

### 2.1 Fine-Grained Classes (16)

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

### 2.3 Why 16 Classes Instead of 3

Training on 16 fine-grained classes and rolling up to 3 health groups outperforms training directly on 3 classes because:

- The model learns **discriminative features** for each food type (burgers look different from pizza)
- Reduces confusion between visually similar healthy/unhealthy items (e.g., grilled chicken vs fried chicken)
- The `not_food` class benefits from explicit negative examples rather than being a catch-all
- Fine-grained predictions are more interpretable for debugging and auditing

---

## 3. Model Architecture

### 3.1 High-Level Pipeline

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

### 3.2 Backbone Options

| Backbone | Feature Dim | Params | Source | When Selected |
|---|---|---|---|---|
| `mobilevit_xxs` | 320 | ~1.3M | timm | CPU fallback (default) |
| `mobilevit_xs` | 384 | ~2.3M | timm | Manual selection |
| `mobilevit_s` | 640 | ~5.6M | timm | Manual selection |
| `vit_b_16` | 768 | ~86M | torchvision / timm | GPU available (default) |
| `vit_b_32` | 768 | ~88M | torchvision / timm | Manual selection |
| `vit_l_16` | 1024 | ~304M | torchvision / timm | Manual selection |

The `auto` backbone setting selects `mobilevit_xxs` on CPU and `vit_b_16` on GPU. For mobile deployment, `mobilevit_xxs` is always used regardless of training backbone.

### 3.3 Temporal Pooling Modes

| Mode | Operation | Description |
|---|---|---|
| `lstm` (default) | `BiLSTM(feat_dim, feat_dim//2, bidirectional=True) -> mean` | Learns temporal frame dependencies |
| `avg` | `feats.mean(dim=1)` | Simple mean pooling |
| `max` | `feats.max(dim=1)` | Max pooling |
| `conv1d` | `Conv1d(feat_dim, feat_dim, kernel=3) -> mean` | Learned temporal convolution |

### 3.4 Input / Output Shapes

| Tensor | Shape | Description |
|---|---|---|
| Input | `(B, 8, 3, 224, 224)` | Batch of videos: 8 frames, RGB, 224x224 |
| Backbone output | `(B*8, feat_dim)` | Per-frame features |
| LSTM output | `(B, 8, feat_dim)` | Contextualized frame features |
| After temporal pool | `(B, feat_dim)` | Video-level features |
| Output logits | `(B, 16)` | Raw class scores |

---

## 4. Dataset

### 4.1 Data Collection

~8,600 images across 16 classes, downloaded from Bing image search via `icrawler`. Each keyword produces ~20 images named with the `_frame_NNNN` convention (pseudo-videos).

- **~27 keywords per class** with diverse search terms (51 for not_food)
- **20 images per keyword** = ~540 images per class
- **not_food** includes hard negatives: empty plates, kitchen utensils, grocery aisles, people, pets

Alternatively, real YouTube videos can be downloaded via `generatedata.py` using `yt-dlp`, then frames are extracted with ffmpeg.

### 4.2 Data Source: HuggingFace

The dataset is hosted on HuggingFace at `maia2000/food-classifier-dataset` so teammates can download it directly instead of scraping. The notebook defaults to HuggingFace download (`USE_HF_DATASET = True`).

### 4.3 Data Augmentation (training only)

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

### 4.4 Data Splitting

Video-level stratified splitting (no frame leakage between splits):

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model training |
| Validation | 15% | Early stopping, LR scheduling |
| Test | 15% | Final evaluation |

The split manifest (`video_split_manifest.json`) ensures all frames from the same video stay in the same split, preventing data leakage.

---

## 5. Training

| Parameter | Value | Why |
|---|---|---|
| Optimizer | AdamW (lr=3e-5, weight_decay=1e-3) | AdamW decouples weight decay; low LR for fine-tuning pretrained backbone |
| Loss | CrossEntropyLoss (label_smoothing=0.1) | Smoothing prevents overconfident predictions, improves generalization |
| Class weighting | Inverse frequency per class | Compensates for imbalanced classes (not_food has more samples) |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients during LSTM training |
| LR schedule | LinearLR warmup (3 epochs) + CosineAnnealingLR (eta_min=1e-7) | Warmup avoids destroying pretrained weights; cosine decay is smooth |
| Early stopping | patience=7, min_delta=5e-5 | Stops training when validation loss plateaus |
| AMP | Enabled on CUDA | Mixed precision for faster training on GPU |
| Epochs | 20 (notebook) / 25 (CLI) | Early stopping usually triggers before max epochs |
| Batch size | 8 | Fits in Colab GPU memory with video frame sequences |
| Dropout | 0.4 | Regularization before the classifier head |

### 5.1 Google Drive Checkpoints

When training on Google Colab, checkpoints are automatically synced to Google Drive (`/content/drive/MyDrive/whispr-checkpoints/`) after each improvement. If the Colab runtime disconnects, training resumes from the last best checkpoint.

---

## 6. Evaluation

- Precision, Recall, F1 (macro-averaged) at fine-grained level -- **do not rely on accuracy alone**
- Confusion matrix (16x16)
- Health-level rollup: precision/recall/F1 and confusion matrix at healthy/unhealthy/not_food level
- K-fold cross-validation (video-grouped, stratified)
- Optional external data testing

---

## 7. Export & Mobile Deployment

### 7.1 Export Formats

| Format | File | Target | Why |
|---|---|---|---|
| TorchScript | `.pt` | Mobile / embedded | Native PyTorch, no extra dependencies |
| ONNX | `.onnx` | Cross-platform (opset 17) | Universal runtime (Android, iOS, web, server) |
| CoreML | `.mlpackage` | iOS 15+ | Apple Neural Engine hardware acceleration |
| TFLite | `.tflite` | Android | Optimized for Android NN API |

### 7.2 Conversion Pipeline

```
best_food_classifier.pth  (PyTorch checkpoint)
  |
  v
[Load model + state dict]
  |
  v
[export_mobile.py]
  |  --format torchscript onnx coreml tflite
  |
  +---> model.pt         (TorchScript, ~5 MB)
  +---> model.onnx       (ONNX opset 17, ~5 MB)
  +---> model.mlpackage  (CoreML, iOS 15+)
  +---> model.tflite     (TFLite via onnx2tf)
```

### 7.3 How to Convert

```bash
# Export all formats
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx coreml tflite

# Export specific formats
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx

# With custom output directory
python export_mobile.py --model models/best_food_classifier.pth --output-dir exported_models/
```

### 7.4 Mobile Inference

On mobile, the model expects:
- **Input:** `float32` tensor of shape `(1, 8, 3, 224, 224)` -- 8 RGB frames, normalized with ImageNet stats
- **Output:** `float32` tensor of shape `(1, 16)` -- logits for each class (apply softmax for probabilities)
- **Preprocessing:** Resize each frame to 224x224, normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

For single-image inference (no video), duplicate the image 8 times to fill the frame sequence.

Example mobile inference flow:

```
1. Capture 8 evenly-spaced frames from video (or duplicate single image x8)
2. Resize each to 224x224
3. Normalize with ImageNet mean/std
4. Stack into tensor [1, 8, 3, 224, 224]
5. Run model (TorchScript / ONNX / CoreML / TFLite)
6. Apply softmax to output [1, 16]
7. Map argmax to class name
8. Map class name to health group via HEALTH_LABELS
```

---

## 8. Project Structure

```
src/vit_video/
  models/vit.py              # MobileViTModel (backbone + BiLSTM + classifier)
  engine/trainer.py           # Training loop, AdamW, LR scheduler, AMP, Drive sync
  data/dataset.py             # VideoDataset, build_dataloaders
  data/splits.py              # Video-level splitting, manifest management
  utils/                      # Hardware, transforms, checkpoint utils
  train.py                    # Training entry point
  test.py                     # Evaluation entry point
  inference.py                # Video/webcam inference
  validate_model.py           # Leakage audit, k-fold CV, external test
  export_mobile.py            # Multi-format model export + model card
  upload_hf.py                # Hugging Face Hub upload
  generatedata.py             # YouTube video download + frame extraction
  run_pipeline.py             # End-to-end pipeline (download -> train -> test -> export)
  paths.py                    # Default directory constants
  _bootstrap.py               # sys.path setup for standalone scripts
  vit_video.ipynb             # Google Colab notebook
  requirements.txt            # Python dependencies
  TECHNICAL.md                # This file
```

---

## 9. Usage

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

# Full pipeline
python run_pipeline.py --dataset-dir food_data --epochs 25

# Google Colab
# Open vit_video.ipynb -- handles everything automatically.
```

---

## 10. HuggingFace Repos

| Repo | Type | Content |
|---|---|---|
| `maia2000/food-classifier-dataset` | dataset | 16-class video frames (`frames/<class>/<video>_frame_NNNN.jpg`) |
| `maia2000/food-classifier` | model | Exported models (TorchScript, ONNX, CoreML, TFLite) |
