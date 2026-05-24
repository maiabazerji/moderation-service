# Analyse d'erreurs — Classifieurs alimentaires

Les deux modèles entraînés :

- **MobileNetV3Small** — 3 classes (`healthy` / `unhealthy` / `not_food`), val acc **89 %**
- **ViT Video** — binaire (`healthy` / `unhealthy`), test acc **72 %**

Cette analyse couvre deux niveaux : les erreurs **dans les données** (qui contaminent la métrique) et les erreurs **des modèles** (mode de défaillance sur la tâche).

---

## A. Erreurs dans les données

### A.1 MobileNetV3 — shortcut Fashion-MNIST

La classe `not_food` était à 100 % composée d'images Fashion-MNIST : 28×28 N&B upsamplées en RGB. Conséquences :

- Le modèle apprend un shortcut trivial : *"image floue grisâtre = not_food"*
- Sur le val set, `not_food` ressort à precision 1.00 / recall 1.00 — chiffre trop beau, sans valeur
- Sur n'importe quelle image couleur HD qui ne contient pas de nourriture (animal, objet, scène, texte), le shortcut ne s'applique plus → misclassification garantie

**Corrigé** : Fashion-MNIST remplacé par CIFAR-10 (10k) + SVHN (10k). À re-mesurer après nouveau run.

### A.2 Mapping santé arbitraire (les deux modèles)

`FOOD101_TO_HEALTH` (MobileNetV3) et `FOOD101_TO_OURS` (ViT) sont codés à la main. Cas litigieux jamais validés :

| classe Food-101 | étiquette actuelle | discutable parce que… |
|---|---|---|
| `pulled_pork_sandwich` | unhealthy | viande maigre rôtie, pas frite |
| `beef_carpaccio` | healthy | riche en lipides, peu de transformation |
| `peking_duck` | healthy | peau frite, plat très gras |
| `risotto` | healthy | beurre + parmesan, calorique |
| `paella` | healthy | varie énormément (fruits de mer vs chorizo) |

**À faire** : faire valider la liste par un nutritionniste, ou au moins 2 personnes en désaccord initial.

### A.3 Pas de test set indépendant

Les deux runs évaluent sur le split de validation, qui sert aussi à monitorer pendant l'entraînement. Donc contamination méthodologique : ce qui est rapporté comme "accuracy" est en partie une mesure d'overfitting au val set.

**À faire** : refaire un split 70/15/15 hash-safe (train / val / test), figer le test, ne l'utiliser qu'à la fin pour publier les chiffres officiels.

### A.4 Biais Food-101 et déséquilibre de classes

- Food-101 est dominé par la cuisine occidentale. Les plats asiatiques, africains, sud-américains sont sous-représentés. Si une classe `pho` ou `ramen` est étiquetée `healthy` mais qu'on n'a que 500 images dessus contre 2000 `hamburger` côté unhealthy, le modèle apprend mieux le côté unhealthy.
- ViT test : 840 `healthy` vs 720 `unhealthy` (ratio 1.17:1). Le class-weighting compense partiellement mais pas assez — une des causes du biais documenté en B.2.

---

## B. Erreurs des modèles

### B.1 MobileNetV3 — les 89 % sont gonflés

Classification report sur le val set (11 200 images) :

| classe | precision | recall | support |
|---|---|---|---|
| healthy | 0.85 | 0.77 | 3 328 |
| not_food | **1.00** | **1.00** | 4 036 |
| unhealthy | 0.81 | 0.88 | 3 836 |

- `not_food` parfait → cf. A.1 (shortcut, pas vraie compétence)
- Vraie accuracy sur la tâche utile (`healthy` vs `unhealthy`) : **~83 %**, pas 89 %
- Biais conservateur : 23 % des plats healthy classés unhealthy (vs 12 % dans l'autre sens). Un poke bowl a 1 chance sur 4 d'être tagué malsain.

### B.2 ViT — sous-prédiction de `unhealthy`

Matrice de confusion (1 560 frames) :

```
                  prédit healthy   prédit unhealthy
vrai healthy           657               183
vrai unhealthy         252               468
```

- 252 vrais `unhealthy` classés `healthy` → 35 % de la classe ratée
- Recall unhealthy : 65 % seulement
- Causes : backbone `mobilevit_xxs` trop petit (1.3 M params), convergence incomplète (4 epochs sur 10 demandées), class-weights mal calibrés

### B.3 ViT — anomalie test > val

Test acc (72 %) > best val acc (64 %). C'est anormal : généralement test ≤ val. Hypothèses :
1. Le checkpoint évalué sur test n'est pas le `best_val_acc` mais le checkpoint final
2. Le split val contient plus d'images ambigües que le test (artefact du `seed=42` sur petit dataset)
3. Fuite de signal (peu probable, splits video-level)

À auditer avant de communiquer les 72 % comme chiffre officiel.

### B.4 Quantization et latence — non mesurées

Le 89 % et le 72 % sont mesurés sur les modèles **float32**. Les artefacts effectivement exportés (TFLite quantisé, TFJS uint8 ~1 MB) ne sont pas évalués. Idem pour la latence d'inférence. Écart potentiel : -1 à -2 pts d'accuracy en uint8.

### B.5 Pas de seuil de confiance

Aucun des deux modèles n'expose un seuil pour basculer en `uncertain`. Tout est forcé en une classe, même les images limites. Il faudrait :

- `healthy` / `unhealthy` quand `max(softmax) > τ` (ex. τ=0.7)
- `uncertain` sinon

**Corrigé côté MobileNetV3** : helper `uncertainty_report()` ajouté dans la cellule 7 qui balaye plusieurs seuils.

---

## Hors périmètre de cette analyse

- Robustesse adversariale
- Diversité culturelle au-delà du constat Food-101 occidental (A.4)
- Asymétrie de coût d'erreur (FN unhealthy plus grave que FN healthy, ou l'inverse ? à arbitrer avant de calibrer les seuils en B.5)
