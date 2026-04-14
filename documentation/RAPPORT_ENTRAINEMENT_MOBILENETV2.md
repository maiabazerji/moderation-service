# Rapport d'entraînement — Classifieur d'images MobileNetV2-0.35

**Projet** : moderation-service / image-classifier
**Ticket** : WHISPR-889
**Branche** : `WHISPR-889/mobilenetv2-training-pipeline`
**Date du rapport** : 14 avril 2026
**Auteur** : Zeyu HOU (HouEpitech)

---

## 1. Résumé exécutif

Ce rapport présente les résultats de la dernière session d'entraînement du classifieur d'images de nourriture, après migration du backbone d'**EfficientNet-B0** vers **MobileNetV2-0.35**. L'objectif principal était de réduire drastiquement la taille du modèle et la latence d'inférence pour un déploiement edge (mobile TFLite, navigateur TFJS), tout en introduisant une classe **« Other »** pour rejeter les images non-alimentaires.

### Résultats clés

| Indicateur | Valeur |
|---|---|
| Accuracy (test) | **84.95 %** |
| F1-score pondéré | **85.08 %** |
| F1-score macro | 83.69 % |
| Taille du modèle (TFLite int) | **0.5 MB** |
| Taille du modèle (TFLite fp16) | 0.8 MB |
| Débit d'inférence (TFLite fp16) | **~356 images/s** |
| Accord Keras vs TFLite-fp16 | 99.69 % |

Le modèle atteint un niveau de performance suffisant pour un déploiement en production edge, avec une empreinte mémoire **~34× plus petite** que le modèle Keras original (17 MB → 0.5 MB).

---

## 2. Contexte et motivation

Le modèle EfficientNet-B0 précédemment utilisé offrait une accuracy de ~92 % sur 8 classes mais pesait trop lourd pour les cibles edge. Le nouveau pipeline :

- Utilise **MobileNetV2 avec alpha = 0.35** (version la plus compacte),
- Ajoute une **9e classe « Other »** alimentée depuis Caltech-101 pour rejeter les entrées non-alimentaires,
- Conserve la taille d'entrée 224×224 pour rester compatible avec le pré-traitement client.

---

## 3. Configuration d'entraînement

### 3.1 Architecture

| Paramètre | Valeur |
|---|---|
| Backbone | `tf.keras.applications.MobileNetV2` (alpha=0.35) |
| Poids initiaux | ImageNet |
| Taille d'entrée | 224 × 224 × 3 |
| Nombre de classes | 9 |
| Tête de classification | GAP → BatchNorm → Dropout(0.3) → Dense(9, softmax, L2=1e-4) |

### 3.2 Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Batch size | 32 |
| Optimiseur | Adam |
| Learning rate (stage 1) | 1e-3 |
| Learning rate (stage 2 / fine-tune) | 2e-5 |
| Epochs configurés (stage 1 / 2) | 30 / 30 |
| Epochs réels (early stopping) | **9 / 6** |
| Loss | `sparse_categorical_crossentropy` |
| EarlyStopping (patience) | 3 |
| ReduceLROnPlateau (patience / factor) | 3 / 0.5 |

### 3.3 Data augmentation

| Transformation | Paramètre |
|---|---|
| RandomFlip | horizontal |
| RandomRotation | 0.15 |
| RandomZoom | 0.2 |
| RandomContrast | 0.2 |
| RandomBrightness | 0.2 |
| RandomTranslation | 0.1 |

### 3.4 Jeu de données

| Split | Nombre d'images |
|---|---|
| Entraînement | 4 497 |
| Validation | 963 |
| Test | 964 |
| **Total** | **6 424** |

**Classes** : Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, **Other**, Pizza, Sandwich.

**Vérification de fuite de données** : aucun chevauchement détecté (0 doublons de chemin, 0 doublons de hash entre train/val/test). Le pipeline utilise un outil dédié de préparation « leak-safe ».

---

## 4. Courbes d'apprentissage

### 4.1 Stage 1 — Entraînement de la tête (backbone gelé)

| Epoch | Val Accuracy | Val Loss | Train Accuracy |
|---|---|---|---|
| 1 | 79.02 % | 0.660 | — |
| 9 (final) | 83.48 % | 0.551 | 83.78 % |
| **Meilleur** | **84.15 %** | 0.531 | — |

L'EarlyStopping s'est déclenché après 9 epochs (sur 30 configurés) avec restauration des meilleurs poids.

### 4.2 Stage 2 — Fine-tuning du backbone complet

| Epoch | Val Accuracy | Val Loss | Train Accuracy |
|---|---|---|---|
| 1 | 84.26 % | 0.558 | — |
| 6 (final) | 84.60 % | 0.506 | 85.71 % |
| **Meilleur** | **84.82 %** | 0.493 | — |

Gain marginal du fine-tuning : +0.67 point d'accuracy. L'écart faible suggère que la tête était déjà proche de l'optimum atteignable avec ce backbone compact.

---

## 5. Résultats sur le jeu de test

### 5.1 Métriques globales

| Métrique | Valeur |
|---|---|
| Accuracy | 84.95 % |
| Précision (macro) | 84.86 % |
| Rappel (macro) | 84.14 % |
| F1-score (macro) | 83.69 % |
| Précision (pondérée) | 86.38 % |
| Rappel (pondéré) | 84.95 % |
| F1-score (pondéré) | 85.08 % |

### 5.2 Performance par classe

| Classe | Précision | Rappel | F1-score | Support |
|---|---|---|---|---|
| Baked Potato | 88.5 % | 78.6 % | 83.2 % | 98 |
| Burger | 90.7 % | 72.3 % | 80.5 % | 94 |
| Crispy Chicken | 83.0 % | 87.4 % | 85.1 % | 95 |
| Donut | 79.1 % | 90.8 % | 84.6 % | 142 |
| Fries | 94.9 % | 78.1 % | 85.7 % | 96 |
| Hot Dog | 92.5 % | 78.7 % | 85.1 % | 94 |
| **Other** | **90.1 %** | **97.3 %** | **93.6 %** | 150 |
| Pizza | 87.9 % | 82.9 % | 85.3 % | 70 |
| Sandwich | 56.9 % | 91.1 % | 70.1 % | 45 |

### 5.3 Observations

- **Other** est la classe la plus fiable (F1 = 93.6 %), ce qui valide la stratégie de classe-rejet pour les images hors-distribution.
- **Sandwich** est clairement la classe la plus faible (précision 56.9 %) : le modèle prédit trop souvent « Sandwich » à tort (rappel élevé, précision basse). Hypothèse : confusion visuelle avec Burger et Crispy Chicken. Support faible (45 images) aggrave le problème.
- **Burger / Fries / Hot Dog** ont une précision élevée mais un rappel plus faible (~75-80 %) : le modèle est prudent sur ces classes et se rabat parfois sur des classes voisines.

---

## 6. Exportations et déploiement edge

Trois formats d'export ont été générés via `tools/convert_model.py` puis validés par `tools/validate_exports.py` sur l'ensemble de test :

| Format | Accuracy | F1 pondéré | Taille | Débit (img/s) |
|---|---|---|---|---|
| Keras (.keras) | 85.58 % | 85.64 % | 5.3 MB | 180 |
| TFLite (int) | 84.75 % | 84.77 % | **0.5 MB** | 257 |
| TFLite fp16 | 85.37 % | 85.43 % | 0.8 MB | **356** |

### 6.1 Cohérence inter-formats

| Comparaison | Accord |
|---|---|
| Keras vs TFLite fp16 | **99.69 %** |
| Keras vs TFLite int | 96.27 % |
| TFLite int vs TFLite fp16 | 96.16 % |

Le format **TFLite fp16** est recommandé pour le déploiement : il offre le meilleur compromis entre taille (0.8 MB), latence (356 img/s) et fidélité au modèle Keras (99.69 % d'accord).

**Note** : les mesures de validation post-export portent sur un jeu évaluation différent du test d'entraînement (964 vs 723 images), d'où les légères différences d'accuracy entre les deux sections.

---

## 7. Comparaison avec la baseline EfficientNet-B0

| Métrique | EfficientNet-B0 (ancien) | MobileNetV2-0.35 (nouveau) | Δ |
|---|---|---|---|
| Accuracy test | 91.40 % | 84.95 % | −6.45 pts |
| F1 pondéré | 91.47 % | 85.08 % | −6.39 pts |
| Nombre de classes | 8 | 9 (+ Other) | — |
| Taille (déployable) | ~17 MB (Keras) | **0.5 MB (TFLite)** | **−97 %** |

**Analyse** : la perte d'accuracy (~6 points) est le prix à payer pour un modèle ~34× plus léger et compatible avec une classe de rejet. Ce compromis est jugé acceptable pour la cible edge, où la contrainte principale est la taille et la latence.

---

## 8. Limitations et pistes d'amélioration

### 8.1 Limitations identifiées

1. **Classe Sandwich sous-représentée** (45 images en test) avec une précision faible (56.9 %).
2. **Fine-tuning peu productif** (+0.67 pt seulement) — possible sous-régularisation ou LR stage 2 sous-optimal.
3. **Classe Donut légèrement sur-prédite** (précision 79.1 % contre un rappel de 90.8 %) : le modèle confond avec Baked Potato / Hot Dog.

### 8.2 Pistes d'amélioration (futurs tickets)

- **Rééquilibrage du dataset** : viser ≥ 200 images par classe en test, et augmenter les samples de Sandwich dans l'entraînement.
- **Augmentation ciblée** : appliquer un `class_weight` ou un sur-échantillonnage des classes faibles.
- **Exploration alpha=0.50** : tester un backbone un peu plus gros (MobileNetV2-0.50) pour voir si +1 MB vaut +2-3 points d'accuracy.
- **Label smoothing** : déjà supporté dans le code (`label_smoothing` dans `model_config`), à activer lors du prochain sweep.
- **Quantization-aware training** (QAT) : pourrait réduire l'écart Keras ↔ TFLite int (96.27 % seulement) en entraînant directement en précision réduite.

---

## 9. Artefacts livrés

| Artefact | Chemin |
|---|---|
| Modèle Keras source | `src/efficientnet_lite_gpu/exports/BestModelEfficientNetLite.keras` |
| TFLite (int) | `src/efficientnet_lite_gpu/exports/tflite/` |
| TFJS | `src/efficientnet_lite_gpu/exports/tfjs/` |
| Config déploiement | `exports/config.json` |
| Labels | `exports/labels.json` |
| Métriques validation | `exports/validation_metrics.json` |
| Rapport HTML d'entraînement | `exports/validation_report.html` |
| Rapport PDF de validation | `exports/rapport_validation_modeles.pdf` |

---

## 10. Conclusion

La migration vers **MobileNetV2-0.35** a atteint ses objectifs de compacité (modèle **~34× plus léger**) et de latence (**~356 img/s en fp16**), au prix d'une baisse d'accuracy de ~6 points par rapport au baseline EfficientNet-B0. L'ajout de la classe **Other** renforce la robustesse face aux entrées hors-distribution (F1 = 93.6 % sur cette classe).

Le modèle est **prêt pour un déploiement edge en pilote**. Les travaux d'amélioration prioritaires concernent le rééquilibrage de la classe Sandwich et l'exploration d'un backbone légèrement plus gros (alpha 0.50) avant de figer la version de production.
