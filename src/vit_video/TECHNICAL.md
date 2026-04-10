# ViT Video Classifier -- Technical Documentation

## Overview

Binary video classification model that labels food content as **healthy** or **unhealthy**. Uses a Vision Transformer (ViT) backbone with temporal pooling to aggregate frame-level features into a single video-level prediction.

**Framework:** PyTorch  
**Task:** Binary classification (healthy / unhealthy)  
**Input:** Video or sequence of frames  
**Output:** Class probabilities (softmax over 2 classes)

---

## 1. Model Architecture

### 1.1 High-Level Pipeline

```
Video (T frames)
  |
  v
[Frame Extraction] -- evenly-spaced sampling (default T=8)
  |
  v
[Per-Frame ViT Backbone] -- pretrained feature extractor
  |  Input:  (B*T, 3, 224, 224)
  |  Output: (B*T, feat_dim)
  v
[Reshape] -- (B, T, feat_dim)
  |
  v
[Temporal Pooling] -- avg / max / conv1d
  |  Output: (B, feat_dim)
  v
[Dropout]
  |
  v
[Linear Classifier] -- (feat_dim -> num_classes)
  |
  v
Output: (B, 2) logits
```

### 1.2 Backbone Options

The model auto-selects the backbone based on available hardware:

| Backbone | Feature Dim | Source | When Selected |
|---|---|---|---|
| `vit_b_16` | 768 | torchvision / timm | GPU available (default) |
| `vit_b_32` | 768 | torchvision / timm | Manual selection |
| `vit_l_16` | 1024 | torchvision / timm | Manual selection |
| `vit_l_32` | 1024 | torchvision / timm | Manual selection |
| `vit_h_14` | 1280 | torchvision / timm | Manual selection |
| `mobilevit_xxs` | 320 | timm | CPU fallback (default) |
| `mobilevit_xs` | 384 | timm | Manual selection |
| `mobilevit_s` | 640 | timm | Manual selection |

**Loading priority:** torchvision first, falls back to timm if unavailable.

All backbones are loaded with `num_classes=0` (feature extractor mode, no classification head). Pretrained on ImageNet.

### 1.3 Temporal Pooling

Three strategies to aggregate frame features across the temporal dimension:

| Mode | Operation | Description |
|---|---|---|
| `avg` (default) | `feats.mean(dim=1)` | Mean pooling over T frames |
| `max` | `feats.max(dim=1)` | Max pooling over T frames |
| `conv1d` | `Conv1d(feat_dim, feat_dim, kernel=3, padding=1) -> mean` | Learned temporal convolution then mean |

### 1.4 Classification Head

```
Dropout(p=0.4)  ->  Linear(feat_dim, 2)
```

### 1.5 Input / Output Shapes

| Tensor | Shape | Description |
|---|---|---|
| Input | `(B, 8, 3, 224, 224)` | Batch of videos: 8 frames, RGB, 224x224 |
| Backbone output | `(B*8, feat_dim)` | Per-frame features |
| After temporal pool | `(B, feat_dim)` | Video-level features |
| Output logits | `(B, 2)` | Raw class scores |

---

## 2. Data Pipeline

### 2.1 Data Collection

Images are downloaded from the web using **Bing image search** (via `icrawler`). Each search keyword produces a set of images named with the `_frame_NNNN` convention so the dataset groups them as pseudo-videos.

**Categories:**
- **Healthy:** 30 keywords (banana, salad, quinoa, salmon, avocado toast, etc.)
- **Unhealthy:** 31 keywords (cheeseburger, pizza, fried chicken, donut, candy, etc.)
- **Images per keyword:** 15

### 2.2 Dataset Class (`VideoDataset`)

Custom PyTorch `Dataset` that handles multiple input formats:

| Input Type | Handling |
|---|---|
| Pre-extracted frames (`*_frame_*.jpg`) | Groups by video stem, samples T evenly-spaced frames |
| Video files (`.mp4`, `.avi`, `.mov`) | Decodes with OpenCV, extracts T evenly-spaced frames |
| Standalone images | Duplicates the image T times |
| Frame directories | Loads frames from subdirectory |

**Frame sampling:**
- Training (augment=True): Random temporal sampling from available frames
- Validation/Test: Deterministic evenly-spaced sampling via `np.linspace`

### 2.3 Data Augmentation

Applied only during training:

| Augmentation | Parameters |
|---|---|
| `RandomResizedCrop` | scale=(0.6, 1.0), ratio=(0.8, 1.2) |
| `RandomHorizontalFlip` | p=0.5 |
| `RandomRotation` | degrees=15 |
| `RandomPerspective` | distortion_scale=0.2, p=0.3 |
| `ColorJitter` | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 |
| `GaussianBlur` | kernel_size=3, sigma=(0.1, 2.0) |
| `RandomErasing` | p=0.3, scale=(0.02, 0.2) |

**Normalization (always applied):**
- Mean: `[0.485, 0.456, 0.406]` (ImageNet)
- Std: `[0.229, 0.224, 0.225]` (ImageNet)

### 2.4 Data Splitting

**Video-level splitting** prevents data leakage (frames from the same video never appear in both train and test):

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model training |
| Validation | 15% | Early stopping, LR scheduling |
| Test | 15% | Final evaluation |

**Strategy:**
- Stratified splitting when each class has >= 2 videos
- Falls back to random splitting for very small datasets
- Splits are persisted in `video_split_manifest.json` for reproducibility
- Manifest is auto-synced when frames on disk change

---

## 3. Training

### 3.1 Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Weight decay | 1e-3 |
| Gradient clipping | max_norm=1.0 |

### 3.2 Loss Function

| Parameter | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Label smoothing | 0.1 |
| Class weighting | Inverse frequency: `total / (num_classes * count_per_class)` |

### 3.3 Learning Rate Schedule

```
Epochs 1-3:   LinearLR warmup (start_factor=0.1)
Epochs 4-25:  CosineAnnealingLR (eta_min=1e-7)
```

Combined via `SequentialLR` with milestone at epoch 3.

### 3.4 Early Stopping

| Parameter | Value |
|---|---|
| Monitor | Validation loss |
| Patience | 7 epochs |
| Min delta | 5e-5 |

### 3.5 Mixed Precision

Automatic Mixed Precision (AMP) is enabled when training on CUDA:
- Forward pass uses `torch.cuda.amp.autocast()` (float16)
- Gradient scaling via `torch.cuda.amp.GradScaler`
- Disabled on CPU/MPS

### 3.6 Checkpointing

Best model saved when validation loss improves. Checkpoint contains:
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "best_val_loss": ...,
    "epoch": ...,
}
```

### 3.7 Default Training Configuration

| Parameter | Default |
|---|---|
| Epochs | 25 |
| Batch size | 8 |
| Frames per video | 8 |
| Image size | 224x224 |
| Backbone | auto (ViT-B/16 on GPU, MobileViT-XXS on CPU) |
| Temporal pool | avg |
| Dropout | 0.4 |
| Class weighting | Enabled |
| Seed | 42 |

### 3.8 Optional LR Search

Lightweight hyperparameter search over learning rate candidates:
- Candidates: `[5e-6, 1e-5, 3e-5, 5e-5, 1e-4]`
- Runs short training (configurable epochs) per candidate
- Selects LR with lowest validation loss

---

## 4. Evaluation

### 4.1 Test Metrics

Computed on the held-out test split:
- **Accuracy** (overall)
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
- **F1 Score** (macro-averaged)
- **Confusion matrix**
- **Per-class precision, recall, F1**
- **Classification report** (scikit-learn)

### 4.2 Model Validation Suite

Three-stage validation to detect overfitting and data leakage:

**1. Data Leakage Audit:**
- Verifies video-level splitting is in place
- Simulates frame-level splitting to measure potential overlap
- Flags class imbalance (ratio > 3x)

**2. K-Fold Cross-Validation:**
- Video-grouped stratified K-fold (default 5 folds, reduced to 3 for notebooks)
- Trains fresh model per fold
- Reports mean/std accuracy and F1 across folds
- Uses: lr=1e-4, weight_decay=1e-3, dropout=0.7, epochs=10

**3. External Data Testing:**
- Downloads fresh videos not in the training set
- Evaluates model on completely unseen data
- Benchmarks:
  - >= 95%: Suspicious (possible overfitting)
  - 85-95%: Realistic
  - 70-85%: Moderate
  - < 70%: Poor generalization

---

## 5. Inference

### 5.1 Video Inference

```python
predict_video(model, video_path, device, transform, num_frames=8, img_size=224)
# Returns: (class_name, confidence, latency_ms)
```

**Pipeline:**
1. Open video with OpenCV
2. Sample `num_frames` evenly-spaced frames
3. Resize each frame to `img_size x img_size`
4. Apply normalization transform
5. Stack into tensor `(1, T, 3, H, W)`
6. Forward pass through model
7. Softmax for probabilities

### 5.2 Webcam Inference

Real-time classification from webcam feed:
- Maintains rolling buffer of `num_frames` frames
- Runs inference on each new frame
- Overlays prediction, confidence, and FPS on video feed
- Press 'q' to quit

---

## 6. Model Export

### 6.1 Export Formats

| Format | File | Target | Notes |
|---|---|---|---|
| TorchScript | `.pt` | Mobile / embedded | `torch.jit.trace()`, optional mobile optimization |
| ONNX | `.onnx` | Cross-platform | opset 17, dynamic batch axes, verified with onnxruntime |
| CoreML | `.mlpackage` | iOS 15+ | Requires `coremltools`, embeds class labels |
| TFLite | `.tflite` | Android | via `ai_edge_torch` or ONNX->TF->TFLite fallback |

### 6.2 Model Card

Auto-generated `model_card.json` with:
- Model name, task, creation date
- Input shape `(B, T, C, H, W)`
- Normalization parameters
- Class names
- Exported formats
- Evaluation metrics (if available)

---

## 7. Hardware Support

| Device | Detection | Optimizations |
|---|---|---|
| CUDA (NVIDIA GPU) | `torch.cuda.is_available()` | cuDNN benchmark, AMP, pin_memory |
| MPS (Apple Silicon) | `torch.backends.mps.is_available()` | Basic support |
| CPU | Fallback | MobileViT backbone selected, num_workers capped at 2 on Windows |

---

## 8. Project Structure

```
src/vit_video/
  models/
    vit.py              # MobileViTModel architecture
  engine/
    trainer.py           # Training loop, optimizer, scheduler
  data/
    dataset.py           # VideoDataset, build_dataloaders
    splits.py            # Video-level splitting, manifest management
  utils/
    hardware.py          # Device detection
    data_utils.py        # Normalization, transforms
    model_utils.py       # Checkpoint loading, backbone detection
    video.py             # Frame extraction from video files
    ytdlp_helpers.py     # YouTube download helpers (legacy)
  generatedata.py        # Data download and frame extraction
  train.py               # Training entry point
  test.py                # Evaluation entry point
  inference.py           # Video/webcam inference
  validate_model.py      # Leakage audit, k-fold CV, external test
  export_mobile.py       # Multi-format model export
  upload_hf.py           # Hugging Face Hub upload
  run_pipeline.py        # End-to-end pipeline orchestrator
  paths.py               # Default directory constants
  _bootstrap.py          # sys.path setup for imports
  vit_video.ipynb        # Google Colab notebook
  requirements.txt       # Python dependencies
```

---

## 9. Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.12
opencv-python>=4.8.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
seaborn>=0.12.0
tqdm>=4.65.0
icrawler (for data download)
onnx>=1.14.0 (optional, for ONNX export)
onnxruntime>=1.15.0 (optional, for ONNX verification)
huggingface-hub>=0.20.0 (optional, for HF upload)
```

---

## 10. Usage

### Training
```bash
python train.py --dataset-dir food_data/frames --epochs 25 --backbone auto
```

### Evaluation
```bash
python test.py --model models/best_food_classifier.pth --dataset-dir food_data/frames
```

### Inference
```bash
python inference.py --video path/to/video.mp4 --model models/best_food_classifier.pth
python inference.py --webcam --model models/best_food_classifier.pth
```

### Export
```bash
python export_mobile.py --model models/best_food_classifier.pth --format torchscript onnx
```

### Full Pipeline
```bash
python run_pipeline.py --epochs 25 --batch-size 8 --backbone auto
```

### Google Colab
Open `vit_video.ipynb` in Colab -- it handles repo cloning, dependency installation, data download, training, evaluation, and export automatically.
