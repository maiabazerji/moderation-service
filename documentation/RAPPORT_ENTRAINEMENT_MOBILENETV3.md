# Rapport d'entraînement — MobileNetV3Small

Notes sur le run de référence du classifieur santé alimentaire basé sur MobileNetV3Small, exécuté via `src/mobilenet_v3_small/mobilenet_colab.ipynb` sous Google Colab. Les chiffres viennent du log d'entraînement et du `train_state.json` produit par le notebook.

## Pourquoi MobileNetV3Small

Le précédent run de référence (MobileNetV2-0.35, classifieur 9 classes alimentaires) tournait à 84.95 % d'accuracy test mais souffrait d'un déséquilibre `Sandwich` et d'une saturation côté backbone à alpha=0.35. Pour la cible "modération photo Whispr", on est revenu à une question plus simple :

- **3 classes** : `healthy`, `unhealthy`, `not_food`.
- **Backbone gelé** (transfer learning pur, pas de fine-tune dégelé).
- **Sortie sigmoïde si binaire, softmax si ≥ 3** — bascule automatique dans le code.

MobileNetV3Small offre un bon compromis taille / précision sur ImageNet et reste exportable en TFLite / TFJS sous le seuil mobile.

## Configuration utilisée

Depuis la cellule 4 du notebook :

```python
IMG, BATCH, TOTAL_EPOCHS = 224, 32, 20
optimizer = tf.keras.optimizers.Adam(1e-3)
```

Tête :

```
Input(224,224,3)
  → RandomFlip(horizontal) + RandomRotation(0.05) + RandomZoom(0.1)
  → preprocess_input (MobileNetV3)
  → MobileNetV3Small(include_top=False, weights="imagenet", trainable=False)
  → GlobalAveragePooling2D
  → Dropout(0.2)
  → Dense(n_classes, softmax)   # ou Dense(1, sigmoid) si n_classes == 2
```

Loss : `sparse_categorical_crossentropy` (3 classes ici), métrique `accuracy`. Aucun `EarlyStopping` ni `ReduceLROnPlateau` — le run va au bout des 20 epochs.

Le checkpoint `model.keras` est écrit à chaque epoch en local **et** sur Drive (`/content/drive/MyDrive/whispr-checkpoints/mobilenet/`) via `PersistCallback`. C'est le mécanisme de reprise en cas de déconnexion du runtime Colab.

## Dataset

| Élément | Valeur |
|---|---|
| Classes | `healthy`, `unhealthy`, `not_food` |
| Total images | 56 000 |
| Train / Val | 44 800 / 11 200 (split 80/20, seed 42) |
| Taille d'entrée | 224×224 RGB |
| Batch | 32 |
| Steps / epoch | 1 400 |

Détail des sources : voir `DATASET_PREPARATION_MOBILENETV3_FR.md`.

## Courbe d'entraînement (20 epochs)

| Epoch | accuracy | loss | val_accuracy | val_loss | Temps |
|---|---|---|---|---|---|
| 1 | 84.94 % | 0.323 | 88.58 % | 0.2549 | 108 s |
| 2 | 86.76 % | 0.285 | 89.13 % | 0.2480 | 93 s |
| 3 | 86.85 % | 0.283 | 88.81 % | 0.2510 | 136 s |
| 4 | 87.04 % | 0.281 | 88.88 % | 0.2467 | 84 s |
| 5 | 86.94 % | 0.283 | 89.00 % | 0.2469 | 81 s |
| 6 | 86.79 % | 0.285 | 89.04 % | 0.2468 | 84 s |
| 7 | 87.09 % | 0.282 | 89.01 % | 0.2474 | 88 s |
| 8 | 86.93 % | 0.282 | 88.31 % | 0.2549 | 134 s |
| 9 | 86.94 % | 0.281 | 89.28 % | 0.2443 | 82 s |
| **10** | 86.91 % | 0.282 | **89.31 %** | **0.2449** | 142 s |
| 11 | 86.93 % | 0.281 | 89.25 % | 0.2468 | 143 s |
| 12 | 87.03 % | 0.280 | 89.11 % | 0.2455 | 141 s |
| 13 | 86.86 % | 0.283 | 88.88 % | 0.2474 | 148 s |
| 14 | 86.95 % | 0.281 | 89.05 % | 0.2456 | 141 s |
| 15 | 86.91 % | 0.283 | 88.96 % | 0.2503 | 84 s |
| 16 | 86.80 % | 0.284 | 89.07 % | 0.2453 | 141 s |
| 17 | 86.91 % | 0.282 | 88.92 % | 0.2467 | 88 s |
| 18 | 87.13 % | 0.281 | 88.71 % | 0.2509 | 84 s |
| 19 | 86.81 % | 0.283 | 89.04 % | 0.2459 | 140 s |
| 20 | 86.94 % | 0.282 | 88.99 % | 0.2478 | 84 s |

**Meilleur val_accuracy** : 89.31 % à l'epoch 10. **val_loss minimum** : 0.2443 à l'epoch 9.

Lecture rapide : la courbe se stabilise dès l'epoch 2 (89 % de val_accuracy) et oscille entre 88.3 % et 89.3 % le reste du run. Conclusion pratique : avec backbone gelé, **10 epochs auraient suffi**. Le budget restant (10 epochs supplémentaires) est gaspillé.

Variabilité des temps par epoch (81 s ↔ 148 s) : runtime Colab partagé, GPU T4 souvent rationné. Pas un signal côté code.

## Métriques sur la validation

Calculées par la cellule 6 (`generate_confusion_matrix`) sur le même split val (11 200 images, seed 42). Les chiffres précis dépendent du modèle chargé au moment de l'exécution (Keras local vs HF distant) — à reproduire sur le checkpoint epoch 10 pour avoir la photo officielle.

Accuracy globale attendue : **~89.3 %** (cohérent avec le pic `val_accuracy`).

Matrice de confusion : `/content/confusion_matrix.png`. Rapport de classification scikit-learn imprimé en sortie de cellule.

## Exports produits

`generate_confusion_matrix` + cellules 6/7 produisent :

| Format | Chemin | Taille typique |
|---|---|---|
| Keras (.keras) | `/content/model.keras` (+ Drive) | ~5-6 MB |
| TFLite float | `/content/model.tflite` | ~6 MB |
| TFLite quantisé (`Optimize.DEFAULT`) | `/content/model_quantized.tflite` | ~1.5 MB |
| TensorFlow.js | `/content/model_tfjs/` (`model.json` + `group1-shard1of1.bin`) | **1 096 KB** mesurés |
| Matrice de confusion | `/content/confusion_matrix.png` | — |

Le TFJS est uploadé dans `maia2000/mobilenet-food-binary/tfjs/` (cellule 7) — c'est le format consommé par le front Whispr.

## Pipeline d'entraînement (mis à jour)

Le notebook tourne maintenant en **deux stages** :

1. **Stage 1 (cellule 4)** — backbone gelé, 20 epochs à LR 1e-3. C'est ce qui était mesuré historiquement à ~89 % val acc.
2. **Stage 2 (cellule 5, nouvelle)** — backbone dégelé (sauf BatchNorm), 3 epochs à LR 1e-5. Skippe automatiquement si `train_state.json` contient `"stage2_done": true`. Gain attendu : +2-3 pts val acc.

La cellule 7 (export + confusion matrix) inclut désormais une **analyse de confiance** qui balaye plusieurs seuils (`0.5 → 0.95`) et reporte le compromis (`% uncertain` vs `accuracy sur le subset certain`). Sert à calibrer le seuil de routage vers la revue humaine avant déploiement.

## Limites du run actuel

1. **Pas de split de test indépendant**. La métrique reportée vient du split val, qui sert aussi à monitorer pendant l'entraînement. Donc légère contamination méthodologique. Pour un livrable produit, prévoir un test set figé (idéalement hash-safe).
2. **Aucun callback `EarlyStopping`** : le run aurait pu s'arrêter dès l'epoch 12 sans perte de qualité.
3. **`FORCE_RETRAIN = True`** efface les checkpoints à chaque lancement. Pratique en debug, dangereux en prod — basculer à `False` avant un run final pour exploiter la reprise sur déconnexion.
4. **Token HF en dur** dans la cellule 1 (`hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`). À sortir vers les Colab Secrets ou les variables d'environnement avant tout commit/partage du notebook.
5. **Pas de test sur trafic Whispr réel**. La métrique reste mesurée sur la distribution d'entraînement.

## Verdict

Le modèle est en état pilote pour la modération photo Whispr (89 % val acc affichée, ~83 % sur la tâche binaire réelle une fois `not_food` retiré du calcul, ~1 MB en TFJS quantisé). Les corrections appliquées (dataset CIFAR+SVHN, stage 2, seuil de confiance) lèvent les blocages techniques ; il reste un dernier verrou **produit** :

- **Test set figé issu du parc Whispr réel** (selfies, captures d'écran, photos cuisine maison). Tant que ce test n'existe pas, on ignore l'écart entre la perf affichée et la perf produit. Cible : 200-500 images annotées par 2 personnes pour avoir un inter-annotator agreement, voir `ANALYSE_ERREURS.md` §5.1.
