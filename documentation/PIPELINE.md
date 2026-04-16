# Pipeline de bout en bout

Note pratique pour traverser l'ensemble du module `src/mobilenet_v2_small/` : depuis les images brutes jusqu'au modèle déployable. Le document suit la façon dont ça se passe réellement aujourd'hui, avec les vraies commandes et les vrais chemins.

Les docs connexes :

- Préparation du dataset → [`DATASET_PREPARATION.md`](./DATASET_PREPARATION.md)
- Sweep d'hyperparamètres → [`HYPERPARAMETER_SWEEP_GUIDE.md`](./HYPERPARAMETER_SWEEP_GUIDE.md)
- Rapport du dernier run MobileNetV2-0.35 → [`RAPPORT_ENTRAINEMENT_MOBILENETV2.md`](./RAPPORT_ENTRAINEMENT_MOBILENETV2.md)
- Environnements GPU → [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md) (laptop) / [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md) (serveur)
- TFJS → [`TFJS_CONVERSION_README.md`](./TFJS_CONVERSION_README.md)

## Avant de commencer

Toutes les commandes ci-dessous s'exécutent depuis `src/mobilenet_v2_small/`.

```bash
cd src/mobilenet_v2_small
python3.11 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
pip install -r requirements.txt
python -m tools.hardware_test           # sanity check GPU
```

Le script `tools/hardware_test.py` appelle `tools_nvidia_cuda.py` — les checks sont orientés NVIDIA et afficheront des warnings inoffensifs sur le serveur AMD. La seule chose qui compte vraiment : `tf.config.list_physical_devices('GPU')` non vide.

## Les deux points d'entrée

Le module a deux façons de lancer un entraînement. Elles existent pour des raisons historiques ; elles ne font pas exactement la même chose.

**`python main.py --action <action>`** — workflow intégré. Passe par `tools/configuration_generator.py` et `tools/config_validator.py`, puis dispatche via le dict `ACTIONS` de `main.py:9` :

```python
ACTIONS = {
    "train": ("train.train", "run"),
    "eval": ("validation.validation", "run"),
    "validation": ("validation.validation", "run"),
    "test": ("test.test", "run"),
}
```

`eval` et `validation` pointent vers le même module — doublon historique. Les flags CLI pour les options de validation se trouvent dans `main.py` (`--validation-image`, `--validation-threshold`, `--validation-model`, `--validation-dataset-dir`, `--validation-no-display`).

**`python -m train.train`** — appelle directement `train/train.py`. Lit `config.yaml`, fait tout le pipeline d'entraînement, sauve le modèle et les rapports. Plus direct, c'est ce que j'utilise quand je ne touche pas à la config.

**`python -m train.run_train`** — script plus ancien, encore présent. Ne lit pas `config.yaml` proprement, il a ses propres constantes en haut de fichier. À garder pour debug quick-and-dirty, à ne pas utiliser pour un run versionné.

## config.yaml, la source de vérité

Tout le pipeline lit `src/mobilenet_v2_small/config.yaml`. Les clés qui comptent vraiment pour un run typique :

```yaml
train_config:
  dataset_dir:      train/dataset_merged        # ce que train.py lit (Train/Val/Test dedans)
  raw_dataset_dir:  train/dataset_raw_cleaned   # source si rebuild_clean_split est actif
  rebuild_clean_split_before_train: true        # si true → split refait au début de train.py
  results_dir:      train/results               # sorties (logs, métriques, plots)

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
    weights: imagenet
    trainable: false                # base gelée en stage 1, libérée en stage 2
    optimizer: adam
    learning_rate: 0.001            # stage 1
    loss: sparse_categorical_crossentropy
    label_smoothing: 0.0            # >0 active CategoricalCrossentropy
    # EarlyStopping et ReduceLROnPlateau dedans aussi
```

Quelques subtilités à connaître :

- `training_divice` (oui, avec le typo) dans `config.yaml` est purement indicatif, le runtime utilise ce que TF détecte.
- `label_smoothing > 0` fait basculer `train.py` en `CategoricalCrossentropy` avec labels one-hot (`train.py:197-209`). Sinon on reste en `sparse_categorical_crossentropy` avec labels `int`.
- `fine_tune_lr` est un défaut de fallback, la vraie valeur utilisée est `stage2_learning_rate` si présente, sinon `fine_tune_lr`, sinon 2e-5 (`train.py:495`).

## Les backbones supportés

Dans `train/train.py:322-328` :

```python
def _get_backbone_class(model_name: str):
    if name in ("efficientnet-b0", "efficientnet_b0"):
        return tf.keras.applications.EfficientNetB0
    elif name in ("efficientnet-b1", "efficientnet_b1"):
        return tf.keras.applications.EfficientNetB1
    elif name in ("efficientnet-b2", "efficientnet_b2"):
        return tf.keras.applications.EfficientNetB2
    elif name in ("efficientnet-b3", "efficientnet_b3"):
        return tf.keras.applications.EfficientNetB3
    elif name in ("mobilenet-v2-035", "mobilenet-v2-050", "mobilenet-v2-100"):
        return tf.keras.applications.MobileNetV2
```

Pour MobileNetV2, le suffixe `-035/050/100` est parsé plus bas (`train.py:500`) pour configurer `alpha=0.35/0.50/1.0`. Ajouter un autre backbone demande deux fonctions (`_get_backbone_class` et `_get_preprocess_input`) et éventuellement la config spécifique du `alpha`/size.

## Un run typique, de bout en bout

### Préparer les données (une seule fois, ou quand les sources changent)

Détails dans [`DATASET_PREPARATION.md`](./DATASET_PREPARATION.md). En résumé :

```bash
python -m tools.fetch_google_dataset
python -m tools.download_other_class --count 800
python -m tools.clean_cross_class_duplicates \
    --input-dir  train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine
```

Si `rebuild_clean_split_before_train: true` dans `config.yaml` (c'est le défaut actif), on saute le `split_dataset.py` manuel — `train.py` s'en charge.

### Entraîner

```bash
python -m train.train
```

Deux phases sous le capot, voir `train.py:_build_and_train_model` :

- **Stage 1** — base gelée, on entraîne la tête (GAP → BN → Dropout → Dense). LR = `learning_rate` de la config (1e-3 aujourd'hui). Budget epochs = `initial_epochs`. `EarlyStopping(patience=3)` et `ReduceLROnPlateau` actifs. Le dernier run a en fait stoppé à 9 epochs sur les 30 budgétés.
- **Stage 2** — toute la base dégelée, LR beaucoup plus petit (2e-5 par défaut). Budget `fine_tune_epochs`. Mêmes callbacks. Le gain observé sur le dernier run : +0.67 pt de `val_accuracy`.

L'augmentation est appliquée **sur les pixels bruts [0, 255]**, puis le preprocess spécifique au backbone (`preprocess_input`) passe derrière — ordre fixé dans `_build_and_train_model` (`train.py:520-523`). Ne pas l'inverser, ça casse la normalisation attendue par ImageNet.

### Ce que le training produit

Dans `train/results/` (chemins contrôlés par `config.yaml`) :

```
train/results/
├── data_exploration/
│   ├── class_distribution.png
│   ├── sample_images.png
│   └── dataset_statistics.png
├── training_logs/
│   ├── training_history.json       # stage1 + stage2, accuracy/loss par epoch
│   └── best_metrics.json           # best val_acc, leak_report, métriques test
├── training_results/
│   ├── training_config.json        # snapshot exact de la config utilisée
│   ├── training_history.json
│   └── training_history.png
└── evaluation_results/
    ├── test_metrics.json
    ├── test_class_report.json      # precision/recall/F1 par classe
    ├── test_class_report.txt       # même chose au format lisible
    ├── confusion_matrix.png
    ├── performance_metrics.png
    ├── class_performance.png
    └── test_confusion_matrix.npy   # matrice brute pour post-process
```

Le modèle lui-même est sauvé à la racine du module sous `BestModelEfficientNetLite.keras`. Le nom est trompeur (rien à voir avec EfficientNet-Lite), il est conservé pour compatibilité avec les consommateurs en aval. On gardera l'alias tant qu'on ne refait pas ces intégrations.

### Évaluer un modèle existant

```bash
python main.py --action eval
```

Rappel : `eval` et `validation` pointent sur le même module. Il y a aussi `--action test` pour un autre script (`test/test.py`) qui fait des prédictions sur un dossier image par image.

### Générer le rapport PDF/HTML

```bash
python -m tools.generate_validation_report
```

Produit `exports/validation_report.html` + `exports/rapport_validation_modeles.pdf` + `exports/validation_metrics.json`.

Le PDF est typiquement livré au PO en fin de sprint. Le HTML est plus confortable pour explorer : charts inter-format, matrices de confusion, performance par classe.

## Export pour l'edge

```bash
python -m tools.convert_model                    # all formats
python -m tools.convert_model --format tflite    # ou tfjs
python -m tools.convert_model --input autre_modele.keras
```

Choix de format dans `tools/convert_model.py` : `all` (défaut), `tflite`, `tfjs`. Il sort :

```
exports/
├── BestModelEfficientNetLite.keras
├── config.json              # {model_type, input_size, preprocessing, class_names, ...}
├── labels.json              # id2label / label2id
├── README.md                # model card simple
├── tflite/
│   ├── model.tflite         # int (~0.5 MB sur MobileNetV2-0.35)
│   └── model_fp16.tflite    # fp16 (~0.8 MB, souvent la meilleure option prod)
└── tfjs/
    ├── model.json
    └── group1-shard*of*.bin
```

Détail important : le modèle d'inférence exporté **retire les couches de data augmentation** (`RandomFlip`, `RandomRotation`, etc.). Elles ne doivent pas être actives en prédiction. Si on voit des prédictions qui varient pour la même image en inference, chercher là en premier.

### Vérifier l'export

```bash
python -m tools.validate_exports
```

Compare Keras vs TFLite vs TFLite-fp16 sur le jeu de test. Seuils dans le script (`validate_exports.py:bottom`) :

- `accuracy > 0.90` → PASS, sinon WARN.
- Accord inter-format : `> 99 %` PASS, `> 95 %` WARN, sinon FAIL.

Chiffres du dernier run (MobileNetV2-0.35) :

- Keras 85.58 %, TFLite 84.75 %, TFLite-fp16 85.37 %.
- Accord Keras ↔ TFLite-fp16 : 99.69 %.
- Accord Keras ↔ TFLite int : 96.27 % — plus bas, on a gardé fp16 pour la prod.

## Publier sur HuggingFace Hub (optionnel)

```bash
huggingface-cli login       # une fois, ou export HF_TOKEN=...
python -m tools.push_to_hub                           # default repo whispr/efficientnet-food-classifier
python -m tools.push_to_hub --repo-id org/autre-nom
python -m tools.push_to_hub --private
python -m tools.push_to_hub --dry-run                 # liste les fichiers sans push
```

Le script pousse tout `exports/` — keras, tflite, tfjs, config.json, labels.json, README. Le README affiché sur HF vient directement de `exports/README.md` ; si on veut un vrai model card riche, l'éditer avant push ou avoir un template versionné.

## Intégration côté client

La règle : toujours se fier à `exports/config.json` pour le preprocessing. Exemple actuel :

```json
{
  "model_type": "mobilenet-v2-035",
  "input_size": 224,
  "input_shape": [1, 224, 224, 3],
  "preprocessing": {
    "resize": [224, 224],
    "normalization": "mobilenet_v2",
    "note": "Use tf.keras.applications.mobilenet_v2.preprocess_input()"
  },
  "class_names": ["Baked Potato", "Burger", ..., "Sandwich"]
}
```

| Cible | Format à utiliser | Charger avec |
|---|---|---|
| Backend Python | `.keras` | `tf.keras.models.load_model(...)` |
| Mobile / Edge | `model_fp16.tflite` | `tf.lite.Interpreter(model_path=...)` |
| Navigateur | `tfjs/model.json` | `tf.loadGraphModel('/…/model.json')` en JS |

Les détails spécifiques TFJS (installation `tensorflowjs`, limites de format) sont dans [`TFJS_CONVERSION_README.md`](./TFJS_CONVERSION_README.md).

## Scénarios qui reviennent souvent

**Je change un hyperparamètre** : éditer `config.yaml`, relancer `python -m train.train`, puis `convert_model` + `validate_exports`. Environ 20-30 min bout en bout sur le serveur AMD (2× RX 6600 XT), plus court si seulement stage 1.

**Je veux comparer plusieurs configs** : utiliser le sweep, `python -m tools.generate_training_report`. Il contient 8 expériences prédéfinies (voir `tools/generate_training_report.py` et [`HYPERPARAMETER_SWEEP_GUIDE.md`](./HYPERPARAMETER_SWEEP_GUIDE.md)). Filtrer avec `--experiments A,B,E`.

**Je change de backbone** : modifier `model_config.model_name` dans `config.yaml`. Pour un backbone pas encore listé dans `_get_backbone_class`, ajouter la paire `_get_backbone_class`/`_get_preprocess_input` dans `train.py`.

**Le modèle en prod ne se comporte pas comme le modèle Keras** :

1. Vérifier le preprocessing côté client (erreur n°1 en pratique).
2. `python -m tools.validate_exports` et regarder l'accord. Si int trop bas, basculer fp16.
3. Vérifier la version du runtime TFLite côté client (mismatch TF 2.15 / tflite-runtime 2.10 → comportements différents sur certains ops rares).

## Une session complète depuis zéro

```bash
cd src/mobilenet_v2_small
source .venv-efficientnet/bin/activate

# Dataset (une seule fois)
python -m tools.fetch_google_dataset
python -m tools.download_other_class --count 800
python -m tools.clean_cross_class_duplicates \
    --input-dir  train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine

# Entraînement + exports + validation
python -m train.train
python -m tools.convert_model
python -m tools.validate_exports
python -m tools.generate_validation_report

# Push (optionnel)
python -m tools.push_to_hub --repo-id whispr/efficientnet-food-classifier
```

Durée observée sur le serveur (ROCm 7.2.1, 2× RX 6600 XT, 9 classes, ~6 400 images) : ~25 min pour le train complet (stage 1 + stage 2), ~90 s pour les exports, ~3 min pour la validation + rapport.
