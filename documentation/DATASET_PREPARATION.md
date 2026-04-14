# Préparation du dataset

Notes pratiques pour préparer le dataset consommé par `src/efficientnet_lite_gpu/train/train.py`. Le document décrit ce que font réellement les scripts sous `tools/`, avec leurs arguments tels qu'ils sont dans le code, pas une réécriture idéalisée.

Toutes les commandes se lancent depuis `src/efficientnet_lite_gpu/`.

## Où va quoi

Les chemins par défaut viennent de `config.yaml` (clés `dataset_dir` et `raw_dataset_dir`) :

```yaml
dataset_dir: train/dataset_merged
raw_dataset_dir: train/dataset_raw_cleaned
```

Le flux réel est :

```
train/dataset_raw_source/        ← images brutes (scrape + Other)
      └─ clean_cross_class_duplicates.py
train/dataset_raw_cleaned/       ← raw_dataset_dir → point d'entrée de train.py
      └─ split_dataset.py  (OU train.py lui-même si rebuild_clean_split_before_train=true)
train/dataset_merged/            ← dataset_dir → ce que l'entraînement lit réellement
   ├─ Train/<Class>/...
   ├─ Val/<Class>/...
   └─ Test/<Class>/...
```

Cas à retenir : dans `config.yaml`, `rebuild_clean_split_before_train: true` est actif. Ça veut dire que `train.py` **refait lui-même** le split hash-safe depuis `raw_dataset_dir` au début de chaque run. Dans ce cas, appeler `split_dataset.py` à la main est redondant — on ne le fait que si on veut voir le rapport ou geler un split donné.

## Sources combinées

Neuf classes au total : 8 alimentaires + `Other` (reject-bin).

| Classe | Source |
|---|---|
| Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Pizza, Sandwich | Scraping web via `tools/fetch_google_dataset.py` |
| Other | Caltech-101 (TFDS) via `tools/download_other_class.py` |

Le nom `Donuts` présent dans la config de scraping est automatiquement mappé vers `Donut` par `CLASS_MERGE_MAP` dans `clean_cross_class_duplicates.py:10` et `split_dataset.py:12` — c'est volontaire, n'y touchez pas sauf si vous voulez ré-introduire la duplication.

---

## 1. Scraper les images alimentaires — `fetch_google_dataset.py`

### Arguments

Vus dans le code (`tools/fetch_google_dataset.py`) :

```
--config <path>   fichier YAML (défaut : tools/dataset_download_config.yaml)
--root   <path>   racine du projet (défaut : parent de tools/)
--dry-run         affiche les catégories et les mots-clés, ne télécharge rien
```

### Config

Le vrai fichier est `tools/dataset_download_config.yaml`. Les clés effectives aujourd'hui :

```yaml
search_engine: duckduckgo         # duckduckgo / bing / google
delay_between_categories: 35      # secondes, évite le rate-limit DDG
delay_after_download: 0.8
delay_between_pages: 2

search_keywords:
  Burger: [...]
  Crispy Chicken: [...]
  Donuts: [...]                   # sera renommé Donut au clean / split
  Fries: [...]
  Pizza: [...]
  Sandwich: [...]

max_num_per_class: 1000
max_rounds_per_class: 5
balance: true
min_size: [200, 200]
# only_classes: [Burger, Fries]   # pour scraper un sous-ensemble
```

DuckDuckGo est conservé comme moteur par défaut parce que Bing/Google rate-limitent plus vite sans clé API. Les délais ne sont pas cosmétiques — les baisser en dessous de ~20 s entre catégories fait cracher des 403 au bout de 2 classes.

### Sortie

```
train/dataset_raw_source/<Classe>/<imgXXXX>.jpg
```

Compter avec bash après coup :

```bash
for c in train/dataset_raw_source/*/; do
    echo "$(basename "$c"): $(find "$c" -type f | wc -l)"
done
```

### Raccourcis Windows

`run_fetch_google_dataset.bat` (téléchargement) et `run_fetch_google_dataset_dry_run.bat` (vérif) à la racine du module — utiles quand on est sur un poste Windows sans `python -m`.

---

## 2. Ajouter la classe `Other` — `download_other_class.py`

### Arguments réels

```
--output      défaut: train/dataset_raw_cleaned/Other
--count       défaut: 700
--image-size  défaut: 224
--seed        défaut: 42
```

**À noter** : le `--output` par défaut pointe dans `dataset_raw_cleaned/`, pas `dataset_raw_source/`. Ce n'est pas une erreur : Caltech-101 est déjà filtré par `FOOD_KEYWORDS` dans le script (`download_other_class.py:25`) pour éviter toute image pouvant ressembler à une classe alimentaire, donc la dédup inter-classes ne lui apporte rien. Le raccourci est assumé.

Si vous voulez rester orthodoxe et faire passer Other par le clean aussi, forcer la sortie :

```bash
python -m tools.download_other_class --output train/dataset_raw_source/Other
```

### Filtrage anti-recouvrement

```python
# download_other_class.py:26
FOOD_KEYWORDS = frozenset({
    "pizza", "food", "sandwich", "burger", "fries",
    "hotdog", "hot_dog", "donut", "doughnut", "potato",
})
```

Tout label Caltech-101 contenant un de ces mots est écarté avant download. Si on élargit la liste des classes alimentaires plus tard, penser à allonger cette frozen-set.

---

## 3. Déduplication inter-classes — `clean_cross_class_duplicates.py`

### Arguments (tous requis)

```
--input-dir       racine contenant <Classe>/<img>...
--output-dir      où écrire la version nettoyée (supprimée si existe déjà !)
--quarantine-dir  où mettre les doublons écartés (supprimée si existe déjà !)
```

### Ce que ça fait vraiment

1. Hash SHA-256 de chaque fichier (`_sha256_file`).
2. Pour chaque hash apparaissant dans plusieurs classes : on **garde une copie** — celle appartenant à la classe **alphabétiquement première** (`clean_cross_class_duplicates.py:62`).
3. Toutes les autres copies (celles des classes non-élues) sont déplacées en quarantaine, pas supprimées.
4. Applique `CLASS_MERGE_MAP` (`Donuts` → `Donut`) au passage.

Conséquence pratique : si Burger et Sandwich partagent une même image, elle restera dans **Burger** (ordre alphabétique) et sera retirée de Sandwich. Ce biais est acceptable tant que les keywords de scraping sont bien distincts.

### Sorties

```
<output-dir>/<Classe>/<img>           # dataset nettoyé
<quarantine-dir>/<Classe>/<img>       # copies rejetées (inspection possible)
<output-dir>/cross_class_cleanup_report.json
```

Extrait d'un report réel :

```json
{
  "unique_hash_kept": 6500,
  "cross_class_duplicates_quarantined": 213,
  "examples": [
    {"hash": "...", "kept_class": "Burger",
     "removed_class": "Sandwich", "removed_path": "..."}
  ]
}
```

### Gotcha

Les deux dossiers `output-dir` et `quarantine-dir` sont **`rmtree`-és** au début s'ils existent (`clean_cross_class_duplicates.py:45-48`). Ne pointez pas `--output-dir` sur un dossier que vous voulez conserver.

---

## 4. Split train/val/test hash-safe — `split_dataset.py`

### Arguments

```
--input-dir        racine type <Classe>/<img>
--output-dir       racine où écrire Train/Val/Test
--train-ratio      0.7
--val-ratio        0.15
--test-ratio       0.15
--mode             copy|move  (défaut copy)
--seed             42
--train-dir-name   Train
--val-dir-name     Val
--test-dir-name    Test
```

Les ratios doivent sommer **strictement** à 1.0 (tolérance `1e-8`, cf `split_dataset.py:27`).

### Ce que ça fait vraiment

- Scanne et hashe chaque image (`scan_raw_dataset`).
- **Si un même hash apparaît dans ≥ 2 classes → exception**, message « Please clean labels first. ». Autrement dit, il faut **passer par le clean en amont** (§3), sinon le split refuse de démarrer.
- Split stratifié **par classe**.
- Les images de même hash sont traitées comme un **groupe indivisible** (`stratified_split`) — elles vont toutes dans le même split, donc aucune fuite possible même si le scraping a produit plusieurs copies dans une même classe.
- Allocation gloutonne (`test` en premier, puis `val`, puis `train`) pour que le ratio cible soit atteint le plus exactement possible.

Applique aussi `CLASS_MERGE_MAP`.

### Sortie

```
<output-dir>/
├── Train/<Classe>/<img>
├── Val/<Classe>/<img>
├── Test/<Classe>/<img>
└── split_report.json   # par-classe / global counts + seed + ratios
```

### Pourquoi ça existe si train.py le refait déjà

Deux cas :

1. Vouloir **figer** un split (et le versionner ou le partager) → lancer à la main.
2. Vouloir **inspecter** le `split_report.json` avant de lancer 40 min d'entraînement.

Pour un run d'entraînement normal avec `rebuild_clean_split_before_train: true`, on peut sauter cette étape.

---

## 5. Recette minimale

```bash
cd src/efficientnet_lite_gpu

# scrape
python -m tools.fetch_google_dataset

# Other
python -m tools.download_other_class --count 800    # ou plus selon besoin

# dédup inter-classes
python -m tools.clean_cross_class_duplicates \
    --input-dir  train/dataset_raw_source \
    --output-dir train/dataset_raw_cleaned \
    --quarantine-dir train/dataset_raw_cross_class_quarantine

# Option A : laisser train.py splitter
python -m train.train     # rebuild_clean_split_before_train: true fait le reste

# Option B : splitter à la main pour geler ou inspecter
python -m tools.split_dataset \
    --input-dir  train/dataset_raw_cleaned \
    --output-dir train/dataset_merged \
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

## 6. État du dataset en production (avril 2026)

Tiré de `train/results/training_logs/best_metrics.json` du dernier run MobileNetV2-0.35 :

| Split | Images |
|---|---|
| Train | 4 497 |
| Val | 963 |
| Test | 964 |

9 classes (`Baked Potato`, `Burger`, `Crispy Chicken`, `Donut`, `Fries`, `Hot Dog`, `Other`, `Pizza`, `Sandwich`). Zéro fuite détectée (hash overlap train/val/test = 0).

## 7. Erreurs vues en vrai

- **`Ratios must sum to 1.0`** : arrondir à 2 décimales exactement, `0.70 + 0.15 + 0.15` (les fractions en `0.7` sans zéro traînant marchent aussi mais j'ai vu des collègues taper `0.7 0.1 0.2` par réflexe).
- **`Found identical image content across different classes`** lors du split : vous n'avez pas passé `clean_cross_class_duplicates` en amont.
- **Classe avec une poignée d'images après scraping** : les quotas DDG sont variables, relancer `--round` plusieurs fois n'est pas toujours suffisant — parfois la classe n'a simplement pas 1000 résultats exploitables, il faut enrichir `search_keywords`.
- **OOM en RAM pendant `clean`** sur de gros volumes : le script tient tout le dict de hashes en mémoire. À 50k fichiers ça passe (~50 Mo de hashes), au-delà il faudrait streamer.

## 8. Ce que j'ai pas traité ici

- **Augmentation** : elle est appliquée au runtime dans la pipeline de modèle (`train.py` dans `_build_data_augmentation`), pas au niveau dataset. Les images sur disque restent intactes.
- **Rééquilibrage par `class_weight`** : n'est pas automatique aujourd'hui. Si une classe est sous-représentée après split (comme `Sandwich` au dernier run, 45 images en test), il faut soit augmenter le quota à l'étape 1, soit ajouter `class_weight` manuellement dans `train.py` — pas encore dans la config.
