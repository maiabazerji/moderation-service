---
license: apache-2.0
tags: [image-classification, food, mobile, mobilenet, tensorflow]
library_name: tensorflow
---

# MobileNetV3 Food Classifier (3-class)

A lightweight, production-ready image classifier for food content moderation. Classifies images into healthy food, unhealthy food, or not-food (non-meal content).

## Model Details

- **Architecture**: MobileNetV3Small (frozen backbone) + trainable classification head
- **Input**: 224x224 RGB images
- **Output**: 3 classes (healthy, unhealthy, not_food)
- **Framework**: TensorFlow/Keras
- **Best Validation Accuracy**: 92.92%

### Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92.92% |
| Precision (macro) | 93.06% |
| Recall (macro) | 93.14% |
| F1-Score (macro) | 93.10% |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|----|
| Healthy | 91% | 93% | 92% | 3,716 |
| Not-Food | 92% | 90% | 91% | 3,000 |
| Unhealthy | 95% | 94% | 95% | 2,998 |

## Training Configuration

### Stage 1: Frozen Backbone (20 epochs)
- ImageNet-pretrained MobileNetV3Small with frozen weights
- Classifier head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(3, softmax)
- Optimizer: Adam (lr=1e-3)
- Loss: Categorical Crossentropy
- Best val accuracy: 91.56%

### Stage 2: Fine-tuning (3 epochs)
- Unfroze backbone, kept 52 BatchNorm layers frozen
- Optimizer: Adam (lr=1e-5)
- Fine-tuned for food-specific feature learning
- Final val accuracy: 92.92%

### Data Augmentation
- RandomFlip (horizontal)
- RandomRotation (0.05)
- RandomZoom (0.1)

## Dataset

- Total: 9,714 validation images from Food-101
- Healthy (3,716): Fruits, grains, salads, seafood, smoothies, soups, vegetables
- Unhealthy (2,998): Burgers, candy, desserts, fried food, pizza, snacks, sugary drinks
- Not-Food (3,000): General objects, street numbers, screenshots

## Model Variants

| Format | Size | Accuracy | Use Case |
|--------|------|----------|----------|
| Dynamic Range | 1.24 MB | 92.65% | Edge devices (optimal) |
| Float16 | 1.96 MB | 92.93% | Mobile apps |
| FP32 | 3.89 MB | 92.93% | Server/inference |

## Uncertainty Analysis

Confidence threshold = 0.70:
- Certain predictions: 90.8%
- Routed to review: 9.2%
- Accuracy on certain: 94.33%

## License

Apache 2.0
