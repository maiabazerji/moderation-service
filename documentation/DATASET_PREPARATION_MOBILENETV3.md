# Préparation du dataset — MobileNetV3Small

Notes pratiques sur la constitution du dataset consommé par `src/mobilenet_v3_small/mobilenet_colab.ipynb`. Le pipeline est ternaire (`healthy`, `unhealthy`, `not_food`) et tourne intégralement sous Google Colab. Tout ce qui suit décrit ce que les cellules font réellement, pas une version idéalisée.

## Vue d'ensemble

Trois étapes successives dans le notebook :

```
HF Hub (maia2000/food-classifier-dataset)
        │  (si vide → fallback)
        ▼
Food-101 (ethz/food101)                       ← images de nourriture, remappées
Fashion MNIST + CIFAR-10 + SVHN               ← images "not_food" (mode, scènes, chiffres)
        ▼
/content/frames/{healthy, unhealthy, not_food}
        ▼
/content/binary/{healthy, unhealthy, not_food}   ← symlinks utilisés par Keras
```

Le classifieur final est entraîné sur les trois sous-dossiers de `/content/binary/`.

## Source principale : HF Hub

```python
HF_DATASET = "maia2000/food-classifier-dataset"
snapshot_download(repo_id=HF_DATASET, repo_type="dataset",
                  local_dir="/content", allow_patterns=["frames/**"])
```

Si le dataset HF est accessible et qu'il contient déjà `frames/**`, le notebook saute le fallback. Le seuil de bascule est codé en dur : `if have < 1000` → on considère que HF a échoué et on enchaîne sur Food-101.

Pratique à retenir : pousser le dataset complet sur HF à la fin d'un run réussi (cellule 5b, `api.upload_large_folder`) évite à un futur runtime de devoir refaire toute la chaîne Food-101 / Fashion-MNIST / CIFAR / SVHN.

## Fallback : Food-101 → binaire santé

Quand HF est vide, on charge `ethz/food101` (75 750 images d'entraînement, 101 classes fines) et on applique un **mapping manuel** `FOOD101_TO_HEALTH` pour replier les classes Food-101 sur deux étiquettes :

- **healthy** : salades (`beet_salad`, `caesar_salad`, `caprese_salad`, `greek_salad`, `seaweed_salad`), poissons et fruits de mer (`grilled_salmon`, `mussels`, `oysters`, `sashimi`, `scallops`, `sushi`, `tuna_tartare`, `shrimp_and_grits`, `crab_cakes`, `lobster_roll_sandwich`), viandes nobles (`filet_mignon`, `peking_duck`, `pork_chop`, `prime_rib`, `steak`), bols complets (`bibimbap`, `paella`, `risotto`), soupes/nouilles (`miso_soup`, `pho`, `ramen`), accompagnements légers (`edamame`, `hummus`, `guacamole`, `deviled_eggs`), plus `ceviche` et `beef_carpaccio` / `beef_tartare`.
- **unhealthy** : fast-food (`hamburger`, `hot_dog`, sandwiches `club`/`grilled_cheese`/`pulled_pork`, `pizza`, `chicken_wings`, `french_fries`, `fried_calamari`, `fried_rice`, `onion_rings`, `spring_rolls`, `fish_and_chips`, `samosa`, `nachos`, `garlic_bread`), pâtisseries (`beignets`, `churros`, `donuts`, `apple_pie`, `baklava`, `bread_pudding`, `cannoli`, `carrot_cake`, `cheesecake`, `chocolate_cake`, `chocolate_mousse`, `creme_brulee`, `cup_cakes`, `panna_cotta`, `red_velvet_cake`, `strawberry_shortcake`, `tiramisu`, `macarons`), petit-déj sucré (`waffles`, `pancakes`, `french_toast`), glaces (`ice_cream`, `frozen_yogurt`).

Toutes les classes Food-101 non listées sont écartées. Le mapping est subjectif (un `pulled_pork_sandwich` reste discutable) mais cohérent à l'échelle binaire.

### Quota par classe source

```python
PER_CLASS_CAP = 500    # par classe Food-101
```

500 images par classe × ~35 classes par catégorie ⇒ ~16-19 k images effectives par cible. Les images sont enregistrées en JPEG qualité 88 directement dans `/content/frames/<healthy|unhealthy>/`.

## Classe `not_food` : objets non-alimentaires

Pour éviter que le modèle prédise `healthy` / `unhealthy` sur n'importe quelle photo, on injecte une troisième classe `not_food` composée d'images neutres :

```python
NOT_FOOD_CAP = 20000
```

Trois sources chargées en cascade jusqu'à atteindre le quota :

| Source | Clé image | Note |
|---|---|---|
| `fashion_mnist` | `image` | 28×28 N&B → converties RGB, upsamplées par Keras au chargement |
| `cifar10` | `img` | 32×32 RGB, scènes naturelles |
| `svhn` (`full_numbers`) | `image` | photos de chiffres de rue, formats variables |

Pratique : Fashion-MNIST suffit à saturer le quota (60 k images dispo), donc CIFAR-10 et SVHN ne se déclenchent que si on monte `NOT_FOOD_CAP` ou si Fashion-MNIST échoue. C'est un choix : la diversité visuelle reste *correcte* à 20 k Fashion-MNIST, mais ajouter CIFAR/SVHN donne un `not_food` plus robuste face aux selfies, captures d'écran et textes.

### Personnalisation

Le notebook imprime explicitement :

> *To further diversify 'not_food' with images of codes, screenshots, selfies, text, you can add your own image files representing these categories to the directory: `/content/frames/not_food`*

Tout `.jpg` déposé dans ce dossier sera pris en compte. C'est l'extension prévue pour les cas réels du produit (modération de photos sur Whispr).

## Organisation finale (cellule 3)

Les fichiers sont consolidés sous `/content/binary/` via **symlinks** (pas de copie) :

```python
BIN = Path("/content/binary")
LABEL_MAP = {**{c: "unhealthy" for c in UNHEALTHY},
             **{c: "healthy"   for c in HEALTHY},
             "not_food": "not_food",
             "healthy": "healthy", "unhealthy": "unhealthy"}  # cas fallback
```

Les listes `UNHEALTHY` / `HEALTHY` codées en cellule 3 (`burgers`, `candy_sweets`, `desserts`, `fried_food`, `pizza`, `salty_snacks`, `sugary_drinks` / `fruits`, `grain_bowls`, `grilled_meat`, `salads`, `seafood`, `smoothies`, `soups`, `vegetables`) ne servent que **si le dataset HF arrive avec ses propres sous-classes**. En mode fallback Food-101, les images sont déjà rangées dans `frames/healthy` ou `frames/unhealthy`, donc seul le `LABEL_MAP.update(...)` final intervient.

À noter : `BIN` est **`rmtree`-é** au début de la cellule. Si vous avez ajouté des images custom directement dans `/content/binary` (et pas dans `/content/frames`), elles sautent au prochain run.

## Bilan d'un run de référence

Comptages observés à la fin de la cellule 2 du run de référence :

| Cible | Images |
|---|---|
| `healthy` | 16 500 |
| `unhealthy` | 19 500 |
| `not_food` | 20 000 |
| **Total** | **56 000** |

Split automatique appliqué par `tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, seed=42)` :

| Split | Images |
|---|---|
| Train | 44 800 |
| Val | 11 200 |

Pas de split de test séparé dans ce notebook — la matrice de confusion (cellule 6) est calculée sur le split `validation`. C'est une limite assumée du run actuel : pour un livrable produit, refaire un split 70/15/15 hash-safe (voir `DATASET_PREPARATION.md` pour la pipeline MobileNetV2) avant publication.

## Pièges déjà rencontrés

- **Téléchargement HF lent** : `HF_HUB_ENABLE_HF_TRANSFER=1` est exporté en cellule 1. Le supprimer rallonge le download Food-101 d'un facteur 3-5.
- **Token HF expiré** → `401 Unauthorized` lors de `api.upload_file` en cellule 5. Vu sur le run de référence. Mettre à jour la valeur de `HF_TOKEN` dans la cellule 1 (ou via les Secrets Colab) avant un nouvel upload.
- **`PIL UserWarning: Truncated File Read`** sur quelques parquets Food-101 : Pillow rattrape l'erreur sur l'image suivante, **aucune action requise**, mais ça pollue le log.
- **Drive non monté** : si l'utilisateur refuse l'autorisation, `DRIVE_DIR` reste `None` et les checkpoints ne sont sauvés que localement (donc perdus à la déconnexion du runtime). Le notebook continue, mais il faut le savoir.

## Ce qui n'est pas géré ici

- **Augmentation d'image** : appliquée au runtime dans le modèle (`tf.keras.Sequential([RandomFlip, RandomRotation(0.05), RandomZoom(0.1)])`), pas sur disque.
- **Rééquilibrage de classes** : aucun `class_weight` passé à `model.fit()`. Les écarts (16 500 vs 20 000) sont absorbés à l'échelle ternaire — pas critique mais à surveiller si on bascule sur du binaire pur.
- **Déduplication inter-sources** : aucune. Food-101 et Fashion-MNIST ne peuvent pas produire d'images identiques en pratique, mais à formaliser si on ajoute d'autres sources d'images alimentaires.
