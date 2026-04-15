# Rapport d'entraînement — MobileNetV2-0.35

Auteur : Zeyu HOU — date : 14 avril 2026.

Notes sur la dernière session d'entraînement du classifieur d'images alimentaires, après la bascule de backbone EfficientNet-B0 → MobileNetV2-0.35 et l'ajout de la classe `Other`. Les chiffres viennent de `train/results/` et `exports/` sur la branche `WHISPR-889/...`.

## Ce qu'on a changé et pourquoi

Le modèle précédent (EfficientNet-B0, 8 classes alimentaires, ~92 % d'accuracy test) était trop lourd pour la cible edge : ~17 MB en Keras, pas de version TFLite viable en dessous de quelques MB. On voulait :

- un modèle qui tient sous 1 MB en TFLite pour le navigateur/mobile ;
- une classe `Other` (non-nourriture) pour éviter les prédictions hautes-confiance sur des entrées aberrantes.

D'où le choix **MobileNetV2 alpha=0.35**, la déclinaison la plus compacte officiellement supportée. On a gardé l'entrée 224×224 pour ne pas casser le preprocessing des consommateurs.

## La config utilisée

Depuis `config.yaml` au moment du run :

- batch 32, 30 epochs budgétées stage 1 et stage 2, LR 1e-3 stage 1 / 2e-5 stage 2.
- `label_smoothing: 0.0` (valeur supportée par le code mais non activée ici).
- Augmentation : RandomFlip horizontal + RandomRotation 0.15 + RandomZoom 0.2 + RandomContrast 0.2 + RandomBrightness 0.2 + RandomTranslation 0.1.
- Tête : `GAP → BN → Dropout(0.3) → Dense(9, softmax, L2=1e-4)`.
- EarlyStopping patience 3 sur `val_loss`, ReduceLROnPlateau patience 3 factor 0.5.

Les backbones autres valides dans le code actuel : `efficientnet-b0..b3`, `mobilenet-v2-035/050/100`.

## Le dataset

Tiré de `best_metrics.json` du run :

- Train / Val / Test : 4 497 / 963 / 964 images, total 6 424.
- 9 classes : Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Other, Pizza, Sandwich.
- Fuite entre splits vérifiée par hash SHA-256 : 0. Tous les scores rapportés sont donc sur un test strictement disjoint de train/val.

## Courbes d'entraînement

Les deux phases ont early-stoppé bien avant les 30 epochs budgétées.

Stage 1 (base gelée, tête seule) :

| Epoch | val_accuracy | val_loss | train_accuracy |
|---|---|---|---|
| 1 | 79.02 % | 0.660 | — |
| 9 (final) | 83.48 % | 0.551 | 83.78 % |
| meilleur | 84.15 % | 0.531 | — |

Stage 2 (base dégelée, fine-tune) :

| Epoch | val_accuracy | val_loss | train_accuracy |
|---|---|---|---|
| 1 | 84.26 % | 0.558 | — |
| 6 (final) | 84.60 % | 0.506 | 85.71 % |
| meilleur | 84.82 % | 0.493 | — |

Le fine-tune n'apporte que +0.67 pt de val_accuracy. Deux hypothèses : soit le LR stage 2 est trop bas et on n'exploite pas assez le dégel, soit la tête a déjà capté l'essentiel de ce que le backbone 0.35 peut offrir et on plafonne. À tester au prochain sweep (voir plus bas).

## Métriques sur le test

| Métrique | Valeur |
|---|---|
| Accuracy | 84.95 % |
| Precision macro | 84.86 % |
| Recall macro | 84.14 % |
| F1 macro | 83.69 % |
| Precision pondérée | 86.38 % |
| Recall pondéré | 84.95 % |
| F1 pondéré | 85.08 % |

Par classe (issu de `test_class_report.json`) :

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baked Potato | 88.5 % | 78.6 % | 83.2 % | 98 |
| Burger | 90.7 % | 72.3 % | 80.5 % | 94 |
| Crispy Chicken | 83.0 % | 87.4 % | 85.1 % | 95 |
| Donut | 79.1 % | 90.8 % | 84.6 % | 142 |
| Fries | 94.9 % | 78.1 % | 85.7 % | 96 |
| Hot Dog | 92.5 % | 78.7 % | 85.1 % | 94 |
| Other | 90.1 % | 97.3 % | 93.6 % | 150 |
| Pizza | 87.9 % | 82.9 % | 85.3 % | 70 |
| Sandwich | 56.9 % | 91.1 % | 70.1 % | 45 |

Ce qui saute aux yeux :

- `Other` marche très bien (F1 = 93.6 %). La stratégie de classe-rejet est validée dans le périmètre actuel.
- `Sandwich` tire la moyenne vers le bas (F1 = 70.1 %, précision 56.9 % seulement). Le modèle sur-prédit "Sandwich" — beaucoup de faux positifs depuis Burger et Crispy Chicken. Support faible (45 images en test) aggrave le diagnostic et fragilise la métrique.
- `Burger / Fries / Hot Dog` ont une précision haute mais un rappel plus bas (~75-80 %) : le modèle reconnaît bien quand il dit "Burger", mais rate des vrais burgers qu'il classe ailleurs (probablement Sandwich).
- `Donut` est dans l'autre sens : rappel élevé (90.8 %), précision plus basse (79.1 %) — il prédit Donut un peu trop souvent, au détriment de Baked Potato et Hot Dog.

## Exports et validation post-conversion

`tools/convert_model.py` sort Keras, TFLite int, TFLite fp16 et TFJS. `tools/validate_exports.py` compare les trois premiers sur le jeu de test :

| Format | Accuracy | F1 pondéré | Taille | Débit |
|---|---|---|---|---|
| Keras | 85.58 % | 85.64 % | 5.3 MB | 180 img/s |
| TFLite int | 84.75 % | 84.77 % | 0.5 MB | 257 img/s |
| TFLite fp16 | 85.37 % | 85.43 % | 0.8 MB | 356 img/s |

Accords inter-formats :

- Keras vs TFLite fp16 : **99.69 %** → PASS (seuil 99 % dans `validate_exports.py`).
- Keras vs TFLite int : 96.27 % → WARN (seuil 95-99 %).
- TFLite int vs fp16 : 96.16 %.

Conclusion pratique : **TFLite fp16** est le format à déployer. 0.8 MB, ~356 img/s en inférence CPU/edge, et quasi indiscernable du Keras sur les prédictions. La version int gagne ~0.3 MB en plus mais l'écart de 3.4 pts d'accord devient sensible sur les classes faibles — pas le bon tradeoff.

Nuance sur les chiffres : l'accuracy affichée dans ce tableau (85.58 %) est supérieure à celle du tableau "Métriques sur le test" (84.95 %) parce que `validate_exports.py` tourne sur un split de test légèrement différent (723 vs 964 images — le script re-construit son propre jeu depuis `train/dataset_merged/Test`). C'est cohérent à 1 % près, pas inquiétant, mais à garder en tête quand on compare.

## Ce qu'on gagne vs EfficientNet-B0

| | B0 (ancien) | MobileNetV2-0.35 (nouveau) | Delta |
|---|---|---|---|
| Test accuracy | 91.40 % | 84.95 % | −6.45 pts |
| F1 pondéré | 91.47 % | 85.08 % | −6.39 pts |
| Classes | 8 | 9 (dont `Other`) | +1 |
| Taille déployable | ~17 MB Keras | 0.5 MB TFLite | −97 % |

On paie ~6 pts d'accuracy pour un modèle ~34× plus léger, avec en bonus une classe de rejet. Sur la cible edge, l'argument principal est la taille et la latence — le tradeoff est tenable. Sur un backend Python côté serveur, il y aurait moins d'intérêt.

## Limites du run actuel

1. **Sandwich sous-représenté**. 45 images en test, précision 56.9 %. Il faut soit plus de données pour Sandwich (cible ≥ 200 en test), soit un `class_weight` dans `model.fit()`. Aucune des deux voies n'est en place.
2. **Fine-tune peu utile** (+0.67 pt). À creuser au sweep : LR stage 2 un peu plus haut (5e-5 ? 1e-4 ?) ou dégeler partiellement le backbone au lieu de tout-ou-rien.
3. **Quantization int imparfaite** (accord 96.27 % vs Keras). Un Quantization-Aware Training donnerait probablement +1-2 pts d'accord, au prix d'un pipeline d'entraînement plus lourd.
4. **Alpha=0.35 pourrait être en dessous du sweet spot**. Tester 0.50 (modèle ~1 MB) au prochain run pour voir si ça vaut 2-3 pts d'accuracy supplémentaires.

Pour mémoire, le code supporte déjà `label_smoothing > 0` (bascule en `CategoricalCrossentropy`) — pas activé ici, candidat pour un prochain sweep.

## Fichiers livrés

Tout est sous `src/efficientnet_lite_gpu/exports/` :

- `BestModelEfficientNetLite.keras` — source Keras (nom legacy).
- `tflite/model.tflite`, `tflite/model_fp16.tflite`.
- `tfjs/model.json` + shards `.bin`.
- `config.json` — métadonnées (input shape, preprocessing, class_names).
- `labels.json` — id2label / label2id.
- `validation_metrics.json` — résultats bruts de `validate_exports.py`.
- `validation_report.html` + `rapport_validation_modeles.pdf` — version lisible, livrable PO.

## Verdict

Le modèle est prêt pour un pilote edge (browser/mobile), pas pour remplacer l'existant côté backend. Deux axes à prioriser avant de figer une v2 :

1. corriger le déséquilibre Sandwich (data ou `class_weight`) ;
2. sweep rapide alpha=0.50 vs 0.35 + LR stage 2 × {5e-5, 1e-4} pour voir si on peut récupérer 2-3 pts sans exploser la taille.
