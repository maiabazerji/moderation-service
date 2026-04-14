# Préparation du dataset

Ce document décrit la chaîne complète de préparation du dataset d'entraînement : **téléchargement → nettoyage → split train/val/test**. L'ensemble produit un dataset prêt pour `train/train.py` sans fuite de données entre splits.

Toutes les commandes sont à lancer depuis `src/efficientnet_lite_gpu/`.

Pour le pipeline global (préparation → entraînement → export), voir [`PIPELINE.md`](./PIPELINE.md).

---

## 0. Vue d'ensemble

```
dataset_raw_source/             ← brut téléchargé (fetch + other)
   ↓  tools.clean_cross_class_duplicates
dataset_raw_cleaned/            ← sans doublons inter-classes
   ↓  tools.split_dataset
dataset/{Train,Val,Test}/       ← prêt pour l'entraînement
```

Chaque étape produit un dossier distinct — la progression est **idempotente** et on peut reprendre à n'importe quelle étape sans perdre le travail précédent.

---

## 1. Sources de données

Le dataset combine **trois sources** :

| Source | Classes | Outil |
|---|---|---|
| Images alimentaires scrapées (DuckDuckGo / Bing / Google) | Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Pizza, Sandwich | `tools.fetch_google_dataset` |
| Classe « Other » (non-nourriture) via Caltech-101 | Other | `tools.download_other_class` |
| Images déjà présentes dans `train/dataset_raw_source/` | — | Ajout manuel possible |

---

## 2. Télécharger les images alimentaires

### 2.1 Configurer les mots-clés

Éditer `tools/dataset_download_config.yaml` :

```yaml
search_engine: duckduckgo         # "duckduckgo" (recommandé), "bing", ou "google"
delay_between_categories: 35      # secondes — anti rate-limit
delay_after_download: 0.8
delay_between_pages: 2

search_keywords:
  Burger: ["burger food", "burger", "hamburger", "cheeseburger", ...]
  Crispy Chicken: ["crispy chicken", "fried chicken", ...]
  Donuts: ["donuts", "donut food", "doughnuts", ...]
  Fries: ["french fries", "fries food", ...]
  Pizza: ["pizza", "margherita pizza", ...]
  Sandwich: ["sandwich", "club sandwich", ...]

max_num_per_class: 1000           # cible par classe
max_rounds_per_class: 5           # relances si < cible
balance: true                     # équilibrer au nombre le plus petit
min_size: [200, 200]              # taille minimum en pixels
# only_classes: [Burger, Fries]   # pour ne télécharger qu'un sous-ensemble
```

### 2.2 Lancer

```bash
# Vérification sèche (valide la config, aucun téléchargement)
python -m tools.fetch_google_dataset --dry-run

# Téléchargement réel
python -m tools.fetch_google_dataset

# Ou avec un fichier de config alternatif
python -m tools.fetch_google_dataset --config tools/dataset_download_config.yaml
```

### 2.3 Scripts Windows (raccourcis)

```bat
run_fetch_google_dataset.bat           :: téléchargement réel
run_fetch_google_dataset_dry_run.bat   :: vérification seule
```

### 2.4 Sortie

```
train/dataset_raw_source/
├── Burger/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── Crispy Chicken/
├── Donuts/
├── Fries/
├── Pizza/
└── Sandwich/
```

> ⚠️ Le scraping peut prendre **plusieurs heures** selon la config (1000 images × 6 classes × délais anti rate-limit). Lancer en `tmux` ou `screen` sur le serveur.

---

## 3. Ajouter la classe « Other »

La classe **Other** agit comme **reject bin** : elle apprend au modèle à dire « ce n'est pas de la nourriture » plutôt que de donner une prédiction haute-confiance sur des images hors-distribution.

### 3.1 Téléchargement depuis Caltech-101

```bash
# Config par défaut (600 images, depuis Caltech-101 via TFDS)
python -m tools.download_other_class

# Plus d'images
python -m tools.download_other_class --count 800

# Dossier de sortie personnalisé
python -m tools.download_other_class --output train/dataset_raw_source/Other
```

Le script **filtre automatiquement** les catégories Caltech-101 qui pourraient chevaucher les classes alimentaires (pizza, sandwich, hot_dog, donut, etc.).

### 3.2 Sortie

```
train/dataset_raw_source/Other/
├── airplane_001.jpg
├── guitar_042.jpg
├── ...
```

---

## 4. Nettoyage — déduplication inter-classes

Les images scrapées peuvent apparaître sous **plusieurs classes** (ex : la même photo renvoyée pour « burger » et « sandwich »). C'est une source majeure de fuite de données si rien n'est fait.

### 4.1 Lancer

```bash
python -m tools.clean_cross_class_duplicates \
    --input-dir  train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine
```

### 4.2 Comportement

- Calcule un **hash SHA-256** de chaque fichier
- Si un même hash apparaît dans ≥ 2 classes → **toutes les copies sont déplacées en quarantaine**, aucune n'est gardée par défaut (sécurité > rappel)
- Applique aussi `CLASS_MERGE_MAP` (ex : `Donuts` → `Donut` pour uniformiser)
- Produit un rapport JSON avec la liste des doublons et la répartition avant/après

### 4.3 Sorties

```
train/dataset_raw_cleaned/                       ← dataset propre (= input moins doublons)
train/dataset_raw_cross_class_quarantine/        ← tous les doublons (inspection manuelle possible)
train/dataset_raw_cleaned/cleanup_report.json    ← statistiques détaillées
```

### 4.4 Vérification rapide

```bash
# Compter les images par classe avant / après
for d in train/dataset_raw_source train/dataset_raw_cleaned; do
    echo "--- $d ---"
    for c in "$d"/*/; do
        echo "$(basename "$c") : $(find "$c" -type f | wc -l)"
    done
done
```

---

## 5. Split train / val / test stratifié et leak-safe

### 5.1 Lancer

```bash
python -m tools.split_dataset \
    --input-dir  train/dataset_raw_cleaned \
    --output-dir train/dataset \
    --train-ratio 0.70 \
    --val-ratio   0.15 \
    --test-ratio  0.15 \
    --seed 42
```

### 5.2 Comportement

- Les ratios doivent sommer à **1.0** exactement (tolérance 1e-8)
- Split **stratifié par classe** (chaque classe respecte les ratios)
- **Hash-based leak check** : aucun fichier (même contenu binaire) n'apparaît dans deux splits
- Applique `CLASS_MERGE_MAP` (ex : `Donuts` → `Donut`)
- Seed fixe pour reproductibilité

### 5.3 Sortie

```
train/dataset/
├── Train/
│   ├── Baked Potato/
│   ├── Burger/
│   ├── Crispy Chicken/
│   ├── Donut/
│   ├── Fries/
│   ├── Hot Dog/
│   ├── Other/
│   ├── Pizza/
│   └── Sandwich/
├── Val/   (même arborescence)
├── Test/  (même arborescence)
└── split_report.json
```

### 5.4 Exemple de `split_report.json`

```json
{
  "seed": 42,
  "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
  "class_counts": {
    "Burger":  {"train": 700, "val": 150, "test": 150, "total": 1000},
    "Pizza":   {"train": 700, "val": 150, "test": 150, "total": 1000},
    "...":     {"..."}
  },
  "leakage_check": {
    "train_val_hash_overlap":  0,
    "train_test_hash_overlap": 0,
    "val_test_hash_overlap":   0
  }
}
```

### 5.5 Vérification indépendante de la fuite

Le pipeline d'entraînement **re-vérifie** l'absence de fuite au démarrage (voir `best_metrics.json → leakage_report`). Les deux vérifications doivent concorder.

---

## 6. Reprendre / rafraîchir le dataset

### 6.1 Ajouter quelques images manuellement

```bash
cp nouvelles_images/*.jpg train/dataset_raw_source/Burger/
# Puis relancer les étapes 4 et 5
```

### 6.2 Re-télécharger une seule classe

Dans `tools/dataset_download_config.yaml`, activer `only_classes: [Burger]` puis :

```bash
python -m tools.fetch_google_dataset
```

### 6.3 Changer les ratios

Il suffit de relancer `tools.split_dataset` — l'output `train/dataset/` est **écrasé** (comportement voulu pour garantir la cohérence de `split_report.json`).

---

## 7. Statistiques du dataset actuel (avril 2026)

Dernière session d'entraînement MobileNetV2-0.35 (voir [`RAPPORT_ENTRAINEMENT_MOBILENETV2.md`](./RAPPORT_ENTRAINEMENT_MOBILENETV2.md)) :

| Split | Nombre d'images |
|---|---|
| Train | 4 497 |
| Val | 963 |
| Test | 964 |
| **Total** | **6 424** |

9 classes : Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, **Other**, Pizza, Sandwich.

**Fuite détectée** : 0 (vérifié par hash SHA-256 sur chaque split).

---

## 8. Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Ratios must sum to 1.0` | Erreur d'arrondi | Utiliser 3 décimales exactes : `0.70 + 0.15 + 0.15` |
| Classe avec < 10 images après split | Scraping DuckDuckGo rate-limité | Réduire `max_num_per_class` ou augmenter `delay_between_pages` |
| `cleanup_report.json` signale 30 %+ de doublons | Plusieurs mots-clés renvoient les mêmes images | Réduire la liste dans `search_keywords` pour chaque classe |
| Fuite détectée par le training | Split refait sans re-nettoyage | Toujours relancer §4 avant §5 si les sources brutes ont changé |
| OOM lors du fetch | Buffer image trop grand | Réduire `max_num_per_class` par batch successifs |

---

## 9. Commandes minimales — reprise complète du dataset

```bash
cd src/efficientnet_lite_gpu

# 1) Scraper les classes alimentaires
python -m tools.fetch_google_dataset

# 2) Ajouter la classe Other
python -m tools.download_other_class --count 800

# 3) Déduplication inter-classes
python -m tools.clean_cross_class_duplicates \
    --input-dir  train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine

# 4) Split stratifié leak-safe
python -m tools.split_dataset \
    --input-dir  train/dataset_raw_cleaned \
    --output-dir train/dataset \
    --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

Le dataset sous `train/dataset/` est alors prêt pour `python -m train.train`.
