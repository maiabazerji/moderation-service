# Préparation du dataset — ViT Video Classifier

Notes pratiques sur la constitution des datasets consommés par `src/vit_video/`. Trois chemins coexistent aujourd'hui :

- **Mode A** — pipeline locale CLI (`generatedata.py` → YouTube + extraction de frames). Mode historique, toujours fonctionnel.
- **Mode B** — notebook multi-classes (`vit_video.ipynb` → snapshot HF + bootstrap Food-101 + scraping Bing). 16 classes fines + rollup santé.
- **Mode C** — notebook binaire **nouvelle version** (`vit_video_binary (2).ipynb` → Food-101 direct depuis HF, remap binaire en mémoire). Aucun frame écrit sur disque, ~10 min sur T4.

Toutes les commandes se lancent depuis `src/vit_video/`.

## Vue d'ensemble

Deux modes coexistent et produisent la même arborescence cible :

```
src/vit_video/food_data/
├── raw_videos/<classe>/*.mp4         ← uniquement en mode YouTube
├── frames/<classe>/*_frame_*.jpg     ← ce que train.py / le notebook lisent
└── video_split_manifest.json         ← split video-level (généré par data/splits.py)
```

Le classifieur est entraîné sur `food_data/frames/<classe>/`. Le fichier `video_split_manifest.json` figure les vidéos (et non les frames) par split, ce qui évite toute fuite frame→vidéo entre train et test.

## Mode A — Pipeline locale CLI (`generatedata.py`)

Mode historique, utilisé pour la version multi-classes du modèle. Télécharge des vidéos YouTube via `yt-dlp` puis extrait des frames espacées uniformément.

### Arguments

```
--dataset-dir           racine du dataset (défaut : src/vit_video/food_data)
--videos-per-keyword    nombre de vidéos téléchargées par mot-clé (défaut 15)
--max-frames-per-video  frames extraits par vidéo (défaut 60)
--frame-size            taille de sortie en pixels (défaut 224)
--min-frames            ignore les vidéos trop courtes (défaut 1, mettre 16 pour filtrer)
--categories-json       remplace le mapping classes/mots-clés par défaut
--extract-workers       parallélisme d'extraction (0 = auto = min(cpu, 8))
```

### Catégories par défaut

Deux super-classes seulement dans `DEFAULT_FOOD_CATEGORIES` (`generatedata.py:26`) : `healthy` (~30 requêtes) et `unhealthy` (~50 requêtes). Le mode multi-classes 16 catégories est défini dans le notebook (`vit_video.ipynb`, cellule 4) et non dans le CLI.

Quelques exemples de mots-clés `healthy` : `green salad bowl meal`, `grilled salmon vegetables plate`, `quinoa bowl healthy meal`, `mediterranean hummus plate`. Côté `unhealthy` : `cheeseburger close up eating`, `pepperoni pizza slice cheese`, `deep fried chicken wings`, `bubble tea boba milk`.

### Pipeline réelle

1. **`yt-dlp`** télécharge `--videos-per-keyword` résultats pour chaque mot-clé via la requête `ytsearch{N}:{query}`. Format préféré : `bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]`, fallback `best[ext=mp4]` si `ffmpeg` indisponible.
2. **`ffmpeg`** (ou `imageio-ffmpeg` en fallback auto-installé) décode les vidéos. Détecté via `_find_ffmpeg`.
3. **`extract_frames_job`** sample `--max-frames-per-video` frames espacées uniformément, redimensionnées à `frame_size`, écrites en JPEG sous `frames/<classe>/<stem>_frame_<NNNN>.jpg`. La nomenclature `*_frame_*.jpg` est cruciale — `VideoDataset` s'en sert pour regrouper les frames par vidéo.
4. **Reprise idempotente** : pour chaque `.mp4`, le script regarde `_existing_frame_stems(output_folder)`. Si le stem est déjà là, la vidéo est sautée. Pratique pour relancer après un échec.

### Sortie

```
food_data/raw_videos/healthy/*.mp4         (~ 30 mots-clés × 15 vidéos ≈ 450 mp4)
food_data/raw_videos/unhealthy/*.mp4       (~ 50 mots-clés × 15 vidéos ≈ 750 mp4)
food_data/frames/healthy/<stem>_frame_*.jpg
food_data/frames/unhealthy/<stem>_frame_*.jpg
```

Volumétrie observée sur un ancien run de référence binaire : **6 377 frames train / 1 140 val** (cf. `exported_models/model_card.json`). Le multiplicateur dépend de la longueur des vidéos et de `--max-frames-per-video`.

## Mode B — Notebook Colab (`vit_video.ipynb`)

Mode utilisé pour la version 16 classes du modèle. Aucun téléchargement YouTube — le notebook combine trois sources d'images statiques.

### B.1 — Snapshot Hugging Face (cellule 5)

```python
HF_DATASET_REPO = "maia2000/food-classifier-dataset"
snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset",
                  local_dir=str(DATASET_DIR),
                  allow_patterns=["frames/**", "video_split_manifest.json"])
```

Si le dépôt HF contient déjà des frames, le notebook saute les étapes suivantes (`downloaded_from_hf = True`). C'est le chemin rapide (~30 s).

### B.2 — Bootstrap Food-101 (cellule 5a)

Quand HF échoue ou est vide, on bootstrappe sur `ethz/food101` :

```python
USE_FOOD101            = True
FOOD101_CAP_PER_SOURCE = 200   # par classe Food-101
FOOD101_SPLIT          = "train"
```

Le mapping `FOOD101_TO_OURS` (cellule 5a) replie les 101 classes Food-101 sur **10 des 16 classes cibles** :

| Classe cible | Sources Food-101 |
|---|---|
| `salads` | `beet_salad`, `caesar_salad`, `caprese_salad`, `greek_salad`, `seaweed_salad` |
| `seafood` | `ceviche`, `grilled_salmon`, `mussels`, `oysters`, `sashimi`, `scallops`, `sushi`, `tuna_tartare`, `shrimp_and_grits`, `crab_cakes`, `lobster_roll_sandwich` |
| `grilled_meat` | `baby_back_ribs`, `beef_carpaccio`, `beef_tartare`, `filet_mignon`, `peking_duck`, `pork_chop`, `prime_rib`, `pulled_pork_sandwich`, `steak`, `foie_gras`, `deviled_eggs` |
| `grain_bowls` | `bibimbap`, `paella`, `risotto` |
| `soups` | `clam_chowder`, `french_onion_soup`, `hot_and_sour_soup`, `lobster_bisque`, `miso_soup`, `pho`, `ramen` |
| `burgers` | `hamburger`, `hot_dog`, `club_sandwich`, `grilled_cheese_sandwich` |
| `pizza` | `pizza` |
| `fried_food` | `chicken_wings`, `french_fries`, `fried_calamari`, `fried_rice`, `onion_rings`, `spring_rolls`, `beignets`, `churros`, `donuts`, `fish_and_chips`, `samosa` |
| `desserts` | `apple_pie`, `baklava`, `bread_pudding`, `cannoli`, `carrot_cake`, `cheesecake`, `chocolate_cake`, `chocolate_mousse`, `creme_brulee`, `cup_cakes`, `panna_cotta`, `red_velvet_cake`, `strawberry_shortcake`, `tiramisu`, `waffles`, `pancakes`, `french_toast`, `macarons` |
| `candy_sweets` | `ice_cream`, `frozen_yogurt` |
| `salty_snacks` | `nachos`, `garlic_bread` |
| `vegetables` | `edamame`, `hummus`, `guacamole` |

Les 6 classes non couvertes par Food-101 (`fruits`, `smoothies`, `sugary_drinks`, `not_food`, `pizza`-extras, etc.) sont peuplées par le scraping Bing en B.3. Les frames Food-101 sont enregistrés avec préfixe `f101_<src>_frame_<NNNN>.jpg` — la nomenclature `_frame_` est respectée pour que `VideoDataset` les indexe correctement.

Cap de 200 images par classe Food-101 × 101 classes ≈ 20 200 images max.

### B.3 — Scraping Bing (cellule 5b) — **désactivée par défaut**

Dans la version courante du notebook (`vit_video.ipynb`), **toute la cellule 5b est commentée**. Le scraping Bing n'est plus lancé automatiquement.

Le code reste présent pour référence ; pour le réactiver, décommenter la cellule et passer :

```python
FORCE_SCRAPE       = True   # forcer le scrape même si HF a marché
IMAGES_PER_KEYWORD = 20
```

Pour chaque classe et mot-clé, `icrawler.BingImageCrawler` télécharge `IMAGES_PER_KEYWORD` images. Les requêtes sont automatiquement suffixées par `" food"` sauf pour `not_food` (cf. `NON_FOOD_CLASSES`). Les fichiers sont renommés en `<slug>_frame_<NNNN>.jpg`. Idempotence : si `cat_dir` contient déjà ≥ `IMAGES_PER_KEYWORD` fichiers pour ce slug, la requête est sautée.

Durée typique : ~30 min pour les 16 classes complètes. **Conséquence pratique de la désactivation** : 3 classes restent vides sur disque dans le run courant (`smoothies`, `sugary_drinks`, `not_food`) — voir §B.6 ci-dessous.

### B.4 — 16 classes définies (cellule 4)

`FOOD_CATEGORIES` liste 16 catégories alimentaires + `not_food` (51 mots-clés réservés à la classe rejet : animaux, sport, paysages, intérieurs sans nourriture). Chaque catégorie contient ~27 mots-clés.

`HEALTH_LABELS` mappe ces 16 classes vers `{healthy, unhealthy, not_food}` pour le rollup d'évaluation (cellule 11).

| Niveau santé | Catégories fines |
|---|---|
| `healthy` | `fruits`, `vegetables`, `salads`, `seafood`, `grilled_meat`, `grain_bowls`, `soups`, `smoothies` |
| `unhealthy` | `burgers`, `pizza`, `fried_food`, `desserts`, `candy_sweets`, `salty_snacks`, `sugary_drinks` |
| `not_food` | `not_food` |

### B.5 — Re-upload du dataset (cellule 6) — **désactivée par défaut**

Comme la 5b, **toute la cellule 6 est commentée** dans la version courante. L'upload HF n'est plus relancé automatiquement (le dépôt `maia2000/food-classifier-dataset` est déjà en place).

Code conservé pour référence ; sortie observée sur le run précédent (14/16 classes peuplées, 15 694 frames) :

```
Uploading 15694 frames from /content/.../food_data/frames -> maia2000/food-classifier-dataset...
[WARN] huggingface_hub: It seems you are trying to upload a large folder at once. ...
```

Pour les très gros datasets, préférer `HfApi().upload_large_folder(...)` (batch commits) à `upload_folder` (single commit).

### B.6 — État réel du dataset sur disque (run de référence courant)

Sortie de la cellule 7 (`cell-inspect`) après bootstrap Food-101 sans scraping Bing :

| Classe | Vidéos uniques | Frames |
|---|---|---|
| `burgers` | 4 | 800 |
| `candy_sweets` | 2 | 400 |
| `desserts` | 18 | 3 600 |
| `fried_food` | 11 | 2 200 |
| `fruits` | 4 | 74 |
| `grain_bowls` | 3 | 600 |
| `grilled_meat` | 11 | 2 200 |
| `pizza` | 1 | 200 |
| `salads` | 5 | 1 000 |
| `salty_snacks` | 2 | 400 |
| `seafood` | 11 | 2 200 |
| `soups` | 7 | 1 400 |
| `vegetables` | 3 | 600 |
| _tmp_download_ | 20 | 20 (rebut) |
| **Total** | | **15 694** |

**Trois classes manquent sur disque** : `smoothies`, `sugary_drinks`, `not_food`. Food-101 ne les couvre pas et le scraping Bing étant désactivé (§B.3), elles restent vides. Conséquences :

- Le split (cellule 8) **ignore les classes vides** — `discover_videos_by_class` ne les liste pas.
- L'entraînement (cellule 9) sera **13-classes en pratique**, pas 16. La constante `NUM_CLASSES` est dérivée du loader (`len(classes)`), donc le modèle s'adapte automatiquement.
- Le `HEALTH_LABELS` rollup (cellule 11) en pâtit : sans `not_food`, le modèle ne peut pas prédire la classe de rejet.

**Action correctrice** : pour combler les trois manquants, soit décommenter la cellule 5b (scraping Bing) et la relancer, soit pousser manuellement des frames dans `food_data/frames/{smoothies,sugary_drinks,not_food}/` avec la nomenclature `<stem>_frame_NNNN.jpg`.

Le sous-dossier `_tmp_download/` (20 frames) est un résidu du scraping précédent — à supprimer avant la cellule 8 (`shutil.rmtree(FRAMES_DIR / "_tmp_download")`), sinon il sera traité comme une 14e classe nommée `_tmp_download`.

## Mode C — Notebook binaire `vit_video_binary (2).ipynb`

Nouveau notebook qui remplace l'ancien `vit_video_binary.ipynb`. Le dataset n'est **pas écrit sur disque** — Food-101 est streamé depuis HF, remappé en binaire en mémoire, puis enveloppé dans des DataLoaders PyTorch via une classe `HFImageDs`.

### C.1 — Source

```python
HF_SOURCE = "ethz/food101"   # 101 classes × 1 000 images train
ds = load_dataset(HF_SOURCE, split="train")    # 75 750 images
```

Aucun fallback. Si HF est inaccessible, le notebook plante immédiatement.

### C.2 — Mapping binaire

Un set explicite `UNHEALTHY` de **58 classes Food-101** (cellule 2) :

```
apple_pie, baby_back_ribs, baklava, beignets, bread_pudding, breakfast_burrito,
cannoli, carrot_cake, cheesecake, chocolate_cake, chocolate_mousse, churros,
club_sandwich, creme_brulee, croque_madame, cup_cakes, donuts, eggs_benedict,
fish_and_chips, french_fries, french_onion_soup, french_toast, fried_calamari,
fried_rice, garlic_bread, grilled_cheese_sandwich, hamburger, hot_dog, ice_cream,
lasagna, lobster_roll_sandwich, macaroni_and_cheese, macarons, nachos,
onion_rings, pancakes, panna_cotta, pizza, poutine, pulled_pork_sandwich,
ravioli, red_velvet_cake, spaghetti_bolognese, spaghetti_carbonara,
strawberry_shortcake, takoyaki, tiramisu, waffles, foie_gras, pork_chop,
cheese_plate, chicken_quesadilla, chicken_wings, clam_chowder, crab_cakes,
frozen_yogurt, lobster_bisque, samosa
```

Tout ce qui n'est pas dans ce set passe en `healthy` (salades, poissons grillés, soupes saines, fruits, viandes maigres, etc.).

### C.3 — Cas frontières discutables

Quelques mappings de cette nouvelle version qui méritent une revue nutritionniste avant un livrable produit :

- `eggs_benedict` → `unhealthy` (œufs + jambon = OK protéique, mais hollandaise = beurre)
- `lobster_roll_sandwich` → `unhealthy` (homard = healthy, mais pain blanc + mayo = unhealthy)
- `crab_cakes` → `unhealthy` (crabe = healthy, mais panure frite = unhealthy)
- `clam_chowder` / `lobster_bisque` → `unhealthy` (poisson dans crème — débat ouvert)
- `cheese_plate` → `unhealthy` (lipides élevés mais peu de transformation)
- `chicken_quesadilla` → `unhealthy` (cheddar + tortilla frite ⇒ OK)
- `pulled_pork_sandwich` → `unhealthy` (sauce barbecue sucrée — OK)
- `spaghetti_bolognese` / `spaghetti_carbonara` → `unhealthy` (pasta + viande/œuf → fait débat)

Différences vs. ancien notebook : le set `UNHEALTHY` est passé d'environ 30 à 58 classes. La frontière s'est durcie côté `unhealthy`.

### C.4 — Split + balance observés

```python
ds = ds.shuffle(seed=42)
n = len(ds); n_val, n_test = int(0.15*n), int(0.15*n)
n_train = n - n_val - n_test
```

Comptages reportés par la cellule 2 :

| Split | Total | healthy | unhealthy |
|---|---|---|---|
| Train | 53 026 | 22 541 | 30 485 |
| Val | 11 362 | 4 861 | 6 501 |
| Test | 11 362 | 4 848 | 6 514 |

Déséquilibre `healthy` / `unhealthy` ≈ 43 / 57 — `unhealthy` majoritaire. Pas de `class_weight` ni de `WeightedRandomSampler` dans la cellule 4 d'entraînement, le modèle hérite donc d'un léger biais vers `unhealthy`.

### C.5 — Transforms PyTorch (cellule 3)

```python
train_tf = Compose([Resize((224,224)), RandomHorizontalFlip(),
                    ColorJitter(0.1, 0.1, 0.1),
                    ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf  = Compose([Resize((224,224)), ToTensor(),
                    Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

Augmentation **beaucoup plus légère** que le multi-classes (`utils/data_utils.py`) : pas de `RandomResizedCrop`, pas de rotation, pas de `GaussianBlur`, pas de `RandomErasing`. Choix cohérent avec le backbone gelé — peu de capacité d'apprentissage côté tête, augmentation forte serait du bruit pur.

Batch size auto-sélectionné : **64 sur CUDA, 8 sur CPU**.

## Split train/val/test — `data/splits.py`

```python
ensure_split_manifest(frames_root=FRAMES_DIR,
                      manifest_path=DATASET_DIR / "video_split_manifest.json",
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      seed=42)
```

Particularité **importante** : le split est **video-level**, pas frame-level. `discover_videos_by_class` regroupe les frames par stem (`<stem>_frame_*.jpg` → vidéo `<stem>`) et c'est la vidéo entière qui va dans un seul split. C'est le seul moyen d'éviter qu'une frame de la vidéo X soit en train et une autre frame de la même vidéo X en val — fuite classique qui gonflerait artificiellement l'accuracy.

Le manifest est figé après création (`ensure_split_manifest` détecte un manifest existant). Pour le régénérer (ajout de vidéos, changement de seed) : `python run_pipeline.py --regenerate-splits --skip-download`.

## Augmentation (au runtime, pas sur disque)

Définie dans `utils/data_utils.py` et appliquée par `VideoDataset` (mode `augment=True`) :

| Transform | Paramètres |
|---|---|
| `RandomResizedCrop` | scale 0.6-1.0, ratio 0.8-1.2 |
| `RandomHorizontalFlip` | p=0.5 |
| `RandomRotation` | 15° |
| `RandomPerspective` | distortion 0.2, p=0.3 |
| `ColorJitter` | brightness/contrast/saturation 0.4, hue 0.1 |
| `GaussianBlur` | kernel 3, sigma 0.1-2.0 |
| `RandomErasing` | p=0.3, scale 0.02-0.2 |

Validation : `Resize(224) + CenterCrop(224)` uniquement. Désactivable via `--disable-augmentation` dans `train.py`.

## Volumétrie de référence

### Multi-classes via notebook (run courant)

Sortie de la cellule 7 (`cell-inspect`) :

| Split | Total frames | Note |
|---|---|---|
| Sur disque | 15 694 | 13 classes peuplées (sur 16) — voir §B.6 |
| Train (70 %) | ~10 986 | calculé par `ensure_split_manifest`, video-level |
| Val (15 %) | ~2 354 | |
| Test (15 %) | ~2 354 | |

3 classes manquantes (`smoothies`, `sugary_drinks`, `not_food`) faussent la comparaison avec les attendus 16-classes.

### Binaire via notebook `vit_video_binary (2).ipynb`

| Split | Total | healthy | unhealthy |
|---|---|---|---|
| Train | 53 026 | 22 541 | 30 485 |
| Val | 11 362 | 4 861 | 6 501 |
| Test | 11 362 | 4 848 | 6 514 |

Source unique : Food-101 (`ethz/food101`, split `train`, 75 750 images), shuffle seed=42, split 70/15/15 image-level (pas de notion de "vidéo" ici, contrairement au multi-classes).

### Ancien run CLI (obsolète)

L'ancien `exported_models/model_card.json` mentionnait Train 6 377 / Val 1 140 / Test 1 560 (`mobilevit_xxs` sur CPU). Ce run est conservé pour traçabilité mais n'est plus représentatif de l'état courant.

## Pièges déjà rencontrés

- **`yt-dlp` rate-limit** sur YouTube : sur certaines vidéos, l'erreur `HTTP Error 429` interrompt le download. Le script catch la `Exception` et passe au mot-clé suivant — pas bloquant mais la classe peut finir sous-représentée.
- **`ffmpeg` absent** : message `[WARN] ffmpeg not available — single-stream mp4 only.` Le fallback `imageio-ffmpeg` est auto-installé via `_ensure_ffmpeg`. Si même ça échoue, le format demandé devient `best[ext=mp4][height<=720]` sans merge audio — fonctionne pour l'extraction de frames (l'audio est ignoré de toute façon).
- **Vidéos trop courtes** : avec `--max-frames-per-video=60` et `--min-frames=1`, on accepte tout. Pour des frames-per-clip à 8 (modèle), passer `--min-frames=16` filtre les clips qui produisent moins de frames qu'un batch temporel.
- **Frames sans préfixe `*_frame_*`** : `VideoDataset` les ignore silencieusement. Si on importe des images d'une autre source (dossier `food_video_dataset_images/`), il faut soit renommer en `<stem>_frame_NNNN.jpg`, soit créer un symlink avec ce schéma.
- **Token HF expiré** côté notebook → cellules 5 et 6 affichent `[SKIP] No HF_TOKEN -- ...`. L'entraînement marche toujours, mais le dataset n'est pas re-poussé.

## Ce qui n'est pas géré ici

- **Dédup inter-vidéos** : deux vidéos YouTube quasi-identiques (re-uploads, mêmes émissions sur deux chaînes) peuvent coexister. Le split video-level protège contre la fuite frame→vidéo mais pas contre la fuite vidéo→vidéo. Si on observe une accuracy val anormalement haute, regarder de ce côté.
- **Class weighting** : géré au runtime (`train.py --class-weighting`), pas au dataset. Les poids sont calculés en inverse-fréquence sur le split train.
- **Filtrage qualité** : aucun. Une frame floue, une vignette de chargement YouTube, ou un cadre noir entre coupures sont tous gardés. À ajouter en pré-processing si on monte la barre.
