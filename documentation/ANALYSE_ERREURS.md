# Analyse d'erreurs — Classifieurs alimentaires

Lecture comparée des modes de défaillance des deux modèles de moderation alimentaire de Whispr :

- **MobileNetV3Small** (3 classes : `healthy` / `unhealthy` / `not_food`, val accuracy ~89.3 %)
- **ViT Video Classifier** (binaire : `healthy` / `unhealthy`, test accuracy 72.12 %)

Les chiffres viennent des runs de référence : `train_state.json` du notebook MobileNetV3 et `src/vit_video/results/test_results.json` + `exported_models/model_card.json`.

Pour la pipeline complète et les hyperparamètres, voir `RAPPORT_ENTRAINEMENT_MOBILENETV3.md` et `RAPPORT_ENTRAINEMENT_VIT.md`.

## 1. Périmètre et méthodologie

| | MobileNetV3 | ViT |
|---|---|---|
| Tâche | Classification image | Classification vidéo (8 frames) |
| Classes | `healthy`, `unhealthy`, `not_food` | `healthy`, `unhealthy` |
| Évaluation | Split val (11 200 images) | Split test indépendant (1 560 frames) |
| Test set externe | Aucun | Aucun (script existe, pas exécuté) |
| Backbone | MobileNetV3Small, gelé | MobileViT-XXS (run CPU), gelé partiel |
| Epochs effectives | 20 / 20 | 4 / 10 |

**Limite méthodologique commune** : aucun des deux runs ne dispose d'un test set issu de la distribution réelle Whispr (selfies, captures d'écran, photos cuisine maison). Les chiffres ci-dessous mesurent la performance **sur la distribution d'entraînement**, pas sur le traffic produit.

## 2. ViT — Analyse fine de la matrice de confusion

Source : `src/vit_video/results/test_results.json` (1 560 frames de test, binaire).

### 2.1 Matrice de confusion

```
                  Prédit
                  healthy   unhealthy
Réel  healthy       657        183       (840, recall 78.2 %)
      unhealthy     252        468       (720, recall 65.0 %)
                   909         651
```

### 2.2 Métriques par classe

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `healthy` | 72.28 % | **78.21 %** | 75.13 % | 840 |
| `unhealthy` | 71.89 % | **65.00 %** | 68.27 % | 720 |

### 2.3 Lecture

**Mode de défaillance dominant : sous-prédiction de `unhealthy`.**

- **252 vrais `unhealthy` classés `healthy`** (35 % des unhealthy) — c'est l'erreur majoritaire en absolu et en relatif.
- **183 vrais `healthy` classés `unhealthy`** (22 % des healthy) — moins grave côté volume.
- Le modèle prédit `healthy` **909 fois** sur 1 560 (58 %) alors que la distribution réelle est 54 %. Biais conservateur de ~4 points.

**Pourquoi ce biais ?**

1. **Déséquilibre support léger** (840 vs 720, ratio 1.17:1). Le class-weighting (poids `[0.9199, 1.0953]` d'après `model_card.json`) compense mais pas assez.
2. **Backbone CPU sous-dimensionné** : `mobilevit_xxs` (~1.3 M params) n'a pas la capacité de capturer les indices fins de `unhealthy` (texture de friture, brillance du sucre, panure). Sur GPU avec `vit_b_16` (~86 M params), l'écart devrait se réduire.
3. **Convergence incomplète** : 4 epochs sur 10 demandées. Le val_loss à 0.95 est encore loin du palier.

### 2.4 Anomalie test > val

Le test accuracy (72.12 %) est supérieur au best val accuracy (63.77 %). C'est **anormal** dans un pipeline sain — généralement test ≤ val. Hypothèses, par ordre de probabilité :

1. **Le modèle évalué sur test n'est pas le `best_val_acc` checkpoint** mais le checkpoint final (epoch 4, après le pic val), ce qui dit que la corrélation val ↔ test est faible sur ce run.
2. **Le split val contient plus de vidéos ambigües** que le test (artefact du `seed=42` sur un dataset de petite taille).
3. **Fuite de signal** : peu probable car le split est video-level (`data/splits.py`), mais à vérifier en lançant `python validate_model.py --model models/best_food_classifier.pth`.

À auditer avant de communiquer les 72 % comme chiffre officiel.

## 3. MobileNetV3 — Inférence des modes de défaillance

Pas de matrice de confusion sauvegardée sur disque (elle vit dans `/content/confusion_matrix.png` côté Colab, pas exportée). On raisonne à partir des courbes d'entraînement et de la composition du dataset.

### 3.1 Indices depuis la courbe

```
Epoch  1: val_acc 88.58 %, val_loss 0.2549
Epoch 10: val_acc 89.31 %, val_loss 0.2449  ← meilleur
Epoch 20: val_acc 88.99 %, val_loss 0.2478
```

Plafonnement très net à partir de l'epoch 2. Trois lectures possibles :

1. **Tâche trop facile** : le dataset est dominé par Fashion-MNIST (20 000 images de vêtements en N&B) qui est trivialement séparable des photos de nourriture. Le modèle gagne 88 % en distinguant simplement "photo de bouffe" vs "le reste", sans vraiment apprendre `healthy` vs `unhealthy`.
2. **Backbone gelé** : aucune adaptation des features ImageNet aux spécificités alimentaires. Un stage 2 fine-tune libérerait probablement 2-3 points.
3. **Tête sous-dimensionnée** : juste `GAP → Dropout(0.2) → Dense(3, softmax)`. Pas d'espace pour modéliser les frontières fines entre `healthy` et `unhealthy`.

### 3.2 Modes de défaillance prévisibles

Vu la composition dataset :

| Type d'erreur | Mécanisme | Volume estimé sur val |
|---|---|---|
| `unhealthy` → `healthy` | Pizza maison, plat frit + salade en arrière-plan | ~3-5 % |
| `healthy` → `unhealthy` | Plat coloré qui ressemble à du fast-food (poke bowl) | ~2-3 % |
| `food` → `not_food` | Photo en gros plan avec texture proche fabric/tissu | ~1-2 % |
| `not_food` → `food` | Bibelot, sticker, magnet de réfrigérateur avec photo de bouffe | ~1-2 % |

Total estimé ~8-12 % d'erreur, cohérent avec le 11 % observé (100 − 89).

**À vérifier en pratique** : ressortir la matrice de confusion en re-lançant la cellule 6 du notebook sur le checkpoint epoch 10 et analyser les misclassifications classe par classe.

### 3.3 Le risque caché : `not_food`

La classe `not_food` est composée à 100 % de **Fashion-MNIST** (vêtements 28×28 N&B upsamplés). Conséquences pratiques :

- Une **photo de t-shirt couleur** ou un **screenshot d'app** ne ressemble à rien de ce que le modèle a vu en `not_food`. Risque de misclassification en `healthy` ou `unhealthy`.
- Le modèle peut apprendre des **shortcuts triviaux** : "image floue 28×28 upsamplée" = `not_food`. Une vraie photo nette dans la même catégorie sémantique tombera ailleurs.

C'est le risque principal côté déploiement Whispr : le `not_food` du dataset ne reflète pas le `not_food` réel (selfies, screenshots, codes, textes).

## 4. Modes de défaillance partagés

Les deux modèles héritent de problèmes structurels communs :

### 4.1 Pas de test externe

Aucun des deux runs n'a été évalué sur des images hors-distribution. La généralisation au traffic Whispr (selfies, captures d'écran, photos cuisine maison sous éclairage variable) est **inconnue**. C'est le risque numéro 1 pour un livrable produit.

Pour ViT : `python validate_model.py --only-external` télécharge des vidéos YouTube fraîches et évalue dessus — pas encore lancé dans ce projet.

Pour MobileNetV3 : le script équivalent n'existe pas, à implémenter ou faire à la main.

### 4.2 Mapping `healthy` / `unhealthy` arbitraire

Les deux modèles utilisent un mapping codé en dur :

- **MobileNetV3** : `FOOD101_TO_HEALTH` dans la cellule 2 du notebook.
- **ViT** : `HEALTH_LABELS` dans la cellule 4 + `FOOD101_TO_OURS` dans la 5a.

Cas frontières discutables et identiques dans les deux mappings :
- `pulled_pork_sandwich` → `unhealthy` (mais c'est de la viande maigre rôtie, pas frite)
- `beef_carpaccio` → `healthy` (viande crue, riche en lipides mais peu de transformation)
- `peking_duck` → `healthy` (peau frite mais conservée comme "viande noble")
- `risotto` → `healthy` (beurre + parmesan, charge calorique haute)
- `paella` → `healthy` (varie énormément, du paella aux fruits de mer au paella au chorizo)

Si le PO veut faire valider le mapping santé par un nutritionniste, c'est la liste à apporter.

### 4.3 Pas de niveau de confiance utilisable

Aucun des deux pipelines n'expose un seuil de confiance pour basculer en `uncertain`. Conséquence : tout est forcé en une classe, même les images limites. Pour la modération, on voudrait :

- `healthy` / `unhealthy` quand `max(softmax) > τ` (ex. τ=0.7)
- `uncertain` sinon, escaladé en revue humaine.

Calibration à ajouter avant déploiement (calibrer τ sur un set de validation représentatif, viser un taux d'`uncertain` < 5 %).

## 5. Recommandations actionables

### 5.1 Avant tout livrable produit

1. **Construire un test set Whispr-représentatif** : 200-500 images du traffic réel, annotées manuellement par 2 personnes pour mesurer l'inter-annotator agreement. C'est la métrique qui compte.
2. **Re-évaluer les deux modèles** sur ce test set. Attendre une chute de 5-10 points par rapport aux chiffres val (89 % MobileNetV3, 72 % ViT) — si la chute est plus forte, c'est un signal de overfitting au dataset synthétique.
3. **Comparer côté `not_food`** : combien de selfies / screenshots passent en `healthy` ou `unhealthy` ? C'est l'erreur la plus visible côté utilisateur.

### 5.2 Correctifs courts (1-2 jours)

| Cible | Action | Gain estimé |
|---|---|---|
| ViT — recall `unhealthy` | Re-run GPU + ViT-B/16 + LSTM + 20 epochs effectives | +5-10 pts recall |
| MobileNetV3 — plafond val | Stage 2 fine-tune (3 epochs, LR 1e-5, backbone dégelé) | +2-3 pts acc |
| MobileNetV3 — `not_food` | Remplacer Fashion-MNIST par un mix CIFAR + SVHN + photos custom | Réduction des FP `not_food` sur images réelles |
| Les deux | Logger les top-k softmax pour identifier les confusions systématiques | Diagnostic, pas un gain direct |

### 5.3 Correctifs longs (1-2 semaines)

- **Re-annoter le mapping santé** avec un nutritionniste ou au moins 2 personnes en désaccord initial, pour figer les frontières (`risotto`, `pulled_pork_sandwich`, `peking_duck`, etc.).
- **Active learning** : déployer le modèle en shadow mode sur le traffic Whispr, logger les prédictions à faible confiance, faire annoter manuellement, ré-injecter dans le dataset.
- **Distillation ViT → MobileNetV3** : si on garde MobileNetV3 pour le navigateur (TFJS) et ViT pour le mobile (PyTorch Mobile Lite), entraîner MobileNetV3 à imiter les sorties de ViT sur les classes ambigües. Permet de propager la capacité du gros modèle vers le petit sans regenérer le dataset.

## 6. Ce qui n'est pas dans cette analyse

- **Vitesse d'inférence** : couvert dans `RAPPORT_ENTRAINEMENT_*.md` côté volumétrie modèle, pas mesurée en latence p99.
- **Robustesse adversariale** : aucune attaque adversariale testée. Pour Whispr, le risque est plutôt l'évasion volontaire (photos floues, angles inhabituels) que le FGSM classique.
- **Biais culturels** : Food-101 est dominé par la cuisine occidentale. Plats asiatiques, africains, sud-américains sous-représentés. À mesurer si la cible est internationale.
- **Coût d'erreur asymétrique** : un FN `unhealthy` (laisser passer une photo de fast-food alors que le user voulait du contrôle alimentaire) n'a pas la même gravité qu'un FN `not_food` (classer une photo de t-shirt comme aliment). À discuter avec PO avant de calibrer les seuils.
