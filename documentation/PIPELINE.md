# Pipeline de bout en bout — du dataset brut au modèle déployé

Ce document décrit le **flux complet** d'utilisation du module `src/efficientnet_lite_gpu/` :
récupération des données → préparation → entraînement → évaluation → conversion → publication / déploiement edge.

Toutes les commandes sont à exécuter depuis `src/efficientnet_lite_gpu/` (sauf mention contraire).

> Pour la préparation détaillée du dataset, voir [`DATASET_PREPARATION.md`](./DATASET_PREPARATION.md).
> Pour l'environnement (CUDA / ROCm), voir [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md) et [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md).
> Pour le sweep d'hyperparamètres, voir [`HYPERPARAMETER_SWEEP_GUIDE.md`](./HYPERPARAMETER_SWEEP_GUIDE.md).

---

## 0. Vue d'ensemble

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ 1. Fetch     │ → │ 2. Clean    │ → │ 3. Split    │ → │ 4. Train     │ → │ 5. Evaluate │
│   + Other    │   │   (dedup)   │   │   train/val │   │   stage1+FT  │   │   test set  │
│   class      │   │             │   │   /test     │   │              │   │             │
└──────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
                                                                                  │
                                                                                  ▼
                                                            ┌──────────────┐   ┌─────────────┐
                                                            │ 7. Publier   │ ← │ 6. Export   │
                                                            │   HF Hub     │   │  TFLite/TFJS│
                                                            └──────────────┘   └─────────────┘
```

Toutes les étapes sont idempotentes et peuvent être rejouées individuellement.

---

## 1. Préparer l'environnement

### 1.1 Selon la machine

| Cible | Guide |
|---|---|
| Laptop développement (NVIDIA RTX 3070, CUDA 12.3) | [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md) |
| Serveur entraînement (AMD RX 6600 XT, ROCm 7.2.1) | [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md) |

### 1.2 Dépendances Python

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
pip install -r requirements.txt
```

### 1.3 Vérification GPU

```bash
python -m tools.hardware_test
```

Doit lister au moins un GPU détecté par TensorFlow.

---

## 2. Récupérer / préparer le dataset

Détails complets : [`DATASET_PREPARATION.md`](./DATASET_PREPARATION.md). Résumé express ci-dessous.

### 2.1 Téléchargement des images alimentaires

```bash
# Depuis src/efficientnet_lite_gpu/
python -m tools.fetch_google_dataset               # toutes les classes
python -m tools.fetch_google_dataset --dry-run     # vérifie la config sans télécharger
```

La config est dans `tools/dataset_download_config.yaml` (mots-clés, quotas, moteur de recherche DuckDuckGo par défaut).

### 2.2 Ajout de la classe « Other » (négatifs)

```bash
python -m tools.download_other_class --count 800
# Produit : train/dataset_raw_source/Other/
```

### 2.3 Déduplication inter-classes

```bash
python -m tools.clean_cross_class_duplicates \
    --input-dir train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine
```

### 2.4 Split train / val / test stratifié, hash-safe

```bash
python -m tools.split_dataset \
    --input-dir train/dataset_raw_cleaned \
    --output-dir train/dataset \
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

Produit `train/dataset/{Train,Val,Test}/<ClassName>/...` sans fuite (vérification par hash SHA-256).

---

## 3. Configurer l'entraînement

Fichier : `config.yaml` à la racine de `src/efficientnet_lite_gpu/`.

Paramètres-clés (valeurs actuelles en production) :

```yaml
train_config:
  batch_size: 32
  image_size: 224
  initial_epochs: 30
  fine_tune: true
  fine_tune_epochs: 30
  fine_tune_lr: 1.0e-5

  data_augmentation:
    randomFlip: horizontal
    randomRotation: 0.15
    randomZoom: 0.2
    randomContrast: 0.2
    randomBrightness: 0.2
    randomTranslation: 0.1

  model_config:
    model_name: mobilenet-v2-035    # ou efficientnet-b0 / mobilenet-v2-050 / 100
    include_top: false
    weights: imagenet
    trainable: false
    optimizer: adam
    output_activation: softmax
    learning_rate: 0.001
    loss: sparse_categorical_crossentropy
    label_smoothing: 0.0
```

Pour changer de backbone : modifier `model_name`. Le code charge automatiquement la classe Keras correspondante et son `preprocess_input`.

---

## 4. Entraîner

### 4.1 Via `main.py` (workflow intégré)

```bash
python main.py --action train
```

### 4.2 Via le module `train` directement

```bash
python -m train.train
```

### 4.3 Script autonome (historique, basé sur `config.yaml` + overrides)

```bash
python -m train.run_train
```

### 4.4 Sorties générées

```
train/results/
├── data_exploration/           # distribution des classes, échantillons, stats
├── training_logs/
│   ├── training_history.json   # courbes stage1 + stage2
│   └── best_metrics.json       # meilleur val_acc, leak-report, test scores
├── training_results/
│   ├── training_config.json    # snapshot de la config utilisée
│   ├── training_history.png
│   └── training_history.json
└── evaluation_results/
    ├── test_metrics.json
    ├── test_class_report.json  # précision / rappel / F1 par classe
    ├── confusion_matrix.png
    ├── performance_metrics.png
    └── class_performance.png
```

Le modèle est sauvegardé sous `BestModelEfficientNetLite.keras` (nom historique, conservé pour compatibilité).

### 4.5 Sweep multi-configs

Pour comparer plusieurs hyperparamètres :

```bash
python -m tools.generate_training_report
# Lance 7-8 expériences définies dans le script, génère un rapport HTML comparatif
```

Voir [`HYPERPARAMETER_SWEEP_GUIDE.md`](./HYPERPARAMETER_SWEEP_GUIDE.md).

---

## 5. Évaluer

### 5.1 Sur le jeu de test

Déjà exécuté automatiquement à la fin de `train.py`. Re-génération manuelle :

```bash
python main.py --action eval
```

### 5.2 Rapport complet (HTML + PDF)

```bash
python -m tools.generate_validation_report
```

Produit :
- `exports/validation_report.html` (interactif)
- `exports/rapport_validation_modeles.pdf` (livrable)
- `exports/validation_metrics.json` (métriques machine-readable)

Voir [`RAPPORT_ENTRAINEMENT_MOBILENETV2.md`](./RAPPORT_ENTRAINEMENT_MOBILENETV2.md) pour un exemple de rapport de run réel.

---

## 6. Exporter pour le déploiement edge

### 6.1 Conversion multi-formats

```bash
# Tout exporter (TFLite + TFLite-fp16 + TFJS)
python -m tools.convert_model

# Format unique
python -m tools.convert_model --format tflite
python -m tools.convert_model --format tfjs
python -m tools.convert_model --input path/to/model.keras
```

Sorties :

```
exports/
├── BestModelEfficientNetLite.keras   # source
├── config.json                        # métadonnées (nom, shape, preprocessing)
├── labels.json                        # id2label / label2id
├── tflite/
│   ├── model.tflite                   # int (0.5 MB)
│   └── model_fp16.tflite              # fp16 (0.8 MB, recommandé production)
└── tfjs/
    ├── model.json
    └── group1-shard*of*.bin
```

Le modèle d'inférence **retire les couches de data augmentation** (elles ne doivent pas être actives en prédiction).

### 6.2 Validation post-export

```bash
python -m tools.validate_exports
```

Compare Keras ↔ TFLite ↔ TFLite-fp16 sur le jeu de test et vérifie :
- Accuracy / F1 par format
- Accord (% de prédictions identiques) entre formats
- Taille des fichiers et débit d'inférence

Voir le rapport détaillé : [`RAPPORT_ENTRAINEMENT_MOBILENETV2.md`](./RAPPORT_ENTRAINEMENT_MOBILENETV2.md) §6.

---

## 7. Publier sur HuggingFace Hub (optionnel)

```bash
# Se logger (une fois)
huggingface-cli login   # ou export HF_TOKEN=xxx

# Pousser
python -m tools.push_to_hub --repo-id whispr/efficientnet-food-classifier
python -m tools.push_to_hub --private   # repo privé
```

Le script pousse le contenu de `exports/` (tous formats + `config.json` + `labels.json` + `README.md`).

---

## 8. Intégration côté client

| Cible | Format | Chargement |
|---|---|---|
| Backend Python | `.keras` | `tf.keras.models.load_model(path)` |
| Mobile / Edge | `model_fp16.tflite` | `tf.lite.Interpreter(model_path=...)` |
| Navigateur Web | `tfjs/model.json` | `tf.loadGraphModel('/path/model.json')` en JS |

**Important** : toujours appliquer le preprocessing indiqué dans `exports/config.json` :

```json
"preprocessing": {
  "resize": [224, 224],
  "normalization": "mobilenet_v2",
  "note": "Use tf.keras.applications.mobilenet_v2.preprocess_input()"
}
```

Pour TFJS, voir [`TFJS_CONVERSION_README.md`](./TFJS_CONVERSION_README.md).

---

## 9. Scénarios typiques

### 9.1 Je change un hyperparamètre et je veux ré-entraîner vite

```bash
# Modifier config.yaml
python -m train.train
python -m tools.convert_model
python -m tools.validate_exports
```

### 9.2 Je veux comparer plusieurs backbones

Éditer `tools/generate_training_report.py` (liste `EXPERIMENTS`) puis :

```bash
python -m tools.generate_training_report --experiments A,B,C
```

### 9.3 J'ai de nouvelles images à intégrer

Ajouter dans `train/dataset_raw_source/<Classe>/` puis relancer §2.3 et §2.4.

### 9.4 Le modèle produit en edge ne se comporte pas comme en Keras

Vérifier avec `tools.validate_exports` — si l'accord Keras ↔ TFLite tombe sous 95 %, soupçonner :
- Preprocessing absent ou incorrect côté client
- Quantization int trop agressive → préférer fp16
- Version TFLite runtime incompatible

---

## 10. Récapitulatif — commandes minimales pour un run complet

```bash
cd src/efficientnet_lite_gpu
source .venv-efficientnet/bin/activate

# Dataset (une seule fois)
python -m tools.fetch_google_dataset
python -m tools.download_other_class --count 800
python -m tools.clean_cross_class_duplicates \
    --input-dir train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine
python -m tools.split_dataset \
    --input-dir train/dataset_raw_cleaned \
    --output-dir train/dataset

# Entraînement + export + validation
python -m train.train
python -m tools.convert_model
python -m tools.validate_exports
python -m tools.generate_validation_report

# Publication (optionnel)
python -m tools.push_to_hub --repo-id whispr/efficientnet-food-classifier
```

Durée typique (serveur ROCm 7.2.1 + RX 6600 XT, ~5000 images, MobileNetV2-0.35) : **~20-30 min** pour l'entraînement complet, **~2 min** pour les exports et validation.
