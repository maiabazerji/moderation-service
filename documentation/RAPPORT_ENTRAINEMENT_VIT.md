# Rapport d'entraînement — ViT Video Classifier

Notes sur les runs de référence du classifieur basé sur Vision Transformer (`src/vit_video/`). Trois runs documentés ici :

- **Run binaire courant** — nouveau notebook `vit_video_binary (2).ipynb`, exécuté sur Colab T4. **Run de référence courant.**
- **Run multi-classes courant** — notebook `vit_video.ipynb`, **dataset préparé mais entraînement non exécuté** dans le snapshot courant.
- **Ancien run CLI** — `train.py` + MobileViT-XXS CPU, conservé pour traçabilité.

Les chiffres viennent des sorties cellules des notebooks et de `src/vit_video/exported_models/model_card.json` / `src/vit_video/results/test_results.json` pour l'ancien run.

## 1. Run binaire — référence courante

### 1.1 Configuration

Issue de `vit_video_binary (2).ipynb`, cellule 4 :

| Paramètre | Valeur |
|---|---|
| Backbone | **ViT-B/16** (torchvision, `ViT_B_16_Weights.IMAGENET1K_V1`) |
| Tête | `nn.Linear(model.hidden_dim, 2)` (768 → 2) |
| Backbone gelé | Oui — tous les paramètres `requires_grad=False`, **seule la tête est entraînée** |
| Optimizer | `AdamW(model.heads.parameters(), lr=1e-3, weight_decay=1e-4)` |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |
| AMP | Activé (`torch.amp.GradScaler("cuda")`) |
| Batch size | 64 (CUDA) ou 8 (CPU) |
| Epochs | 10 max, EarlyStopping `patience=3` sur `val_acc` |
| Image size | 224 × 224 |
| Normalisation | ImageNet |

Tâche : binaire (`healthy` vs `unhealthy`). Pas de classe `not_food` dans ce pipeline.

### 1.2 Dataset

| Split | Total | healthy | unhealthy |
|---|---|---|---|
| Train | 53 026 | 22 541 | 30 485 |
| Val | 11 362 | 4 861 | 6 501 |
| Test | 11 362 | 4 848 | 6 514 |

Source : Food-101 (`ethz/food101`), 101 classes fines remappées vers binaire (cellule 2, 58 classes en `UNHEALTHY` explicite, tout le reste en `healthy`). Détail dans `DATASET_PREPARATION_VIT.md` §Mode C.

Pas de class-weighting actif (déséquilibre ~43/57 ignoré).

### 1.3 Courbe d'entraînement (10 epochs)

| Epoch | train_loss | train_acc | val_loss | val_acc |
|---|---|---|---|---|
| 1 | 0.5028 | 78.81 % | 0.4897 | 80.36 % |
| 2 | 0.4829 | 80.52 % | 0.4888 | 80.36 % |
| 3 | 0.4799 | 80.93 % | 0.4830 | 80.90 % |
| 4 | 0.4787 | 81.03 % | 0.4823 | 80.91 % |
| 5 | 0.4778 | 81.27 % | 0.4828 | 80.93 % |
| 6 | 0.4775 | 81.05 % | 0.4861 | 80.87 % |
| 7 | 0.4779 | 81.08 % | 0.4816 | 81.09 % |
| **9** | 0.4780 | 81.02 % | 0.4818 | **81.17 %** |
| 10 | 0.4785 | 81.03 % | 0.4822 | 80.96 % |

**Meilleur val_accuracy** : **81.17 %** à l'epoch 9. Pas d'early-stop déclenché (patience 3, mais le pic à 9 puis baisse à 10 met l'arrêt à l'epoch 12 ce qui n'arrive jamais — le notebook s'arrête simplement à `range(1, 11)`).

Lecture : la courbe plafonne après l'epoch 3. Le backbone gelé impose un plafond ~81 % — la tête linéaire (768 → 2) n'a pas la capacité d'aller plus loin sur la séparation Food-101 binaire. Pour passer ce palier, il faudrait dégeler les 2-3 derniers blocs du transformer (LR ~1e-5, AdamW sur l'ensemble), pas activé ici.

### 1.4 Métriques sur le test

Sortie de la cellule 5 (`c-eval`), évaluée sur le checkpoint `best.pth` (epoch 9) :

| Métrique | Valeur |
|---|---|
| **Accuracy** | **80.88 %** |
| Macro F1 | 80.33 % |
| ROC-AUC | **88.54 %** |

Rapport par classe (`classification_report`) :

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `healthy` | 79.02 % | **75.14 %** | 77.04 % | 4 848 |
| `unhealthy` | 82.15 % | **85.16 %** | 83.63 % | 6 514 |
| Macro avg | 80.59 % | 80.15 % | 80.33 % | 11 362 |
| Weighted avg | 80.82 % | 80.88 % | 80.81 % | 11 362 |

**Mode de défaillance dominant** : sous-prédiction de `healthy` (recall 75 %). 25 % des vrais `healthy` (1 207 frames) sont classés `unhealthy`. Inversion par rapport à l'ancien run CLI (où c'était `unhealthy` qui était sous-prédit). Cause probable : déséquilibre de support (4 848 vs 6 514, ratio 1.34:1 en faveur de `unhealthy`) sans correction par `class_weight`.

L'AUC à 88.54 % indique une bonne séparabilité des probabilités — le modèle ordonne bien les exemples par confiance, mais le seuil 0.5 est sous-optimal. Calibrer un seuil à ~0.45-0.48 réduirait l'écart precision/recall sur `healthy`.

### 1.5 Exports

| Format | Taille | Note |
|---|---|---|
| TorchScript `.pt` | **335 MB** | Trace + `optimize_for_mobile` |
| PyTorch Mobile Lite `.ptl` | **335 MB** | `_save_for_lite_interpreter`, format mobile canonique |
| ONNX `.onnx` | 101 KB | ⚠ Anormalement petit — vérifier si les poids sont externalisés ou si l'export a échoué silencieusement |
| Best PyTorch `.pth` | ~343 MB | Checkpoint state_dict complet |

**Anomalie ONNX** : un ViT-B/16 entier devrait peser ~330 MB en ONNX. Un fichier de 101 KB indique que **seul le graphe d'opérations a été exporté, sans les poids** (option `external_data=True` activée par défaut sur opset 18, ou bug d'export). À ré-exporter avec `torch.onnx.export(..., dynamo=False)` ou vérifier la présence d'un `.onnx_data` sidecar.

Vérification de parité réussie côté `.ptl` : `np.allclose(torch_out, lite_out, atol=1e-4)` passe.

### 1.6 Tentative TFLite échouée (cellule 8)

La dernière cellule du notebook tente une conversion `TorchScript → ONNX → TF SavedModel → TFLite` avec `SELECT_TF_OPS`. **Elle échoue** avec :

```
ValueError: Exporting a ScriptModule is not supported.
Maybe try converting your ScriptModule to an ExportedProgram using
`TS2EPConverter(mod, args, kwargs).convert()` instead.
```

Cause : `torch.onnx.export` (nouveau backend `dynamo=True` par défaut sur torch 2.10) refuse les `ScriptModule`. La cellule charge `model.torchscript.pt` puis essaie de le re-exporter en ONNX — la chaîne `.pt → ONNX → TF → TFLite` ne tient pas debout.

**Workaround à implémenter** : re-faire l'export ONNX **depuis le modèle non-tracé** (`torch.jit.load` → `model.eval()`, puis recharger les poids dans un `vit_b_16` Python natif, puis `torch.onnx.export(...)`). Ou plus simple : utiliser `ai_edge_torch` directement sur le modèle PyTorch, comme le multi-classes (`export_mobile.py`).

Comme documenté dans `src/vit_video/TECHNICAL.md` §4a, **TFJS n'est de toute façon pas viable** pour un ViT à cause de la chaîne de conversion. Pour le navigateur, basculer sur MobileNetV3.

### 1.7 Publication HF

Cellule 7 : push automatique vers **`maia2000/food-classifier-binary-vit`** (model card, `best.pth`, `model.torchscript.pt`, `model_mobile.ptl`, `model.onnx`, `metrics.json`, `confusion.png`).

## 2. Run multi-classes — état courant

Le notebook `vit_video.ipynb` (16 classes) a été partiellement exécuté dans le snapshot courant :

- Cellules 1-3 (setup, install, HF login) — OK
- Cellules 4-7 (config + bootstrap Food-101) — 15 694 frames sur 13 classes (cf. `DATASET_PREPARATION_VIT.md` §B.6)
- Cellule 8 (split manifest) — généré
- **Cellules 9-13 (training, eval, exports) — non exécutées dans ce snapshot**

Pas de chiffres d'entraînement ou de test disponibles pour cette version. Pour produire un run multi-classes officiel :

1. Combler les 3 classes manquantes (`smoothies`, `sugary_drinks`, `not_food`) — soit décommenter la cellule 5b (Bing), soit pousser des frames manuellement.
2. Nettoyer le résidu `_tmp_download/` qui contient 20 frames mal classés.
3. Lancer les cellules 9-13 sur Colab GPU. Wall-clock estimé : 3-4 h sur T4 avec ViT-B/16 + BiLSTM.

Hyperparamètres prévus (cellule 9, inchangés) :

| Paramètre | Valeur |
|---|---|
| Optimizer | AdamW, weight_decay=1e-3 |
| LR | 3e-5 |
| Dropout (tête) | 0.4 |
| Gradient clipping | `max_grad_norm=1.0` |
| EarlyStopping | patience=7, min_delta=5e-5 |
| Tête temporelle | `lstm` (BiLSTM) |
| Backbone | `auto` (résolu en `vit_b_16` sur GPU) |
| Epochs | 20 |
| Class weighting | Activé (inverse-fréquence) |

Différence notable avec la version précédente : `EXPORT_FORMATS` inclut maintenant **`'tfjs'`** par défaut (cellule 13), avec une note interne précisant que cela requiert MobileViT-XXS pour produire un bundle web-déployable (~1.5 MB uint8) — ViT-B/16 produirait un bundle ~330 MB, non viable navigateur. Voir `src/vit_video/TECHNICAL.md` §4a.

Le repo HF cible est passé de `maia2000/food-classifier` à **`maia2000/food-classifier-vitb16`** (cellule 14).

## 3. Ancien run CLI (obsolète)

Pour traçabilité. Documenté dans `exported_models/model_card.json`.

| Paramètre | Valeur |
|---|---|
| Backbone | `mobilevit_xxs` (run CPU) |
| Mode | Binaire (`healthy`, `unhealthy`) |
| Frames par clip | 8 |
| Tête temporelle | `avg` |
| Class weighting | Activé, poids `[0.9199, 1.0953]` |
| Train / Val / Test | 6 377 / 1 140 / 1 560 frames |
| Epochs effectives | 4 / 10 (interruption précoce) |
| Best val accuracy | 63.77 % |
| Test accuracy | 72.12 % |

Dépassé par le run binaire courant (Test acc 80.88 %, dataset 8× plus gros, ViT-B/16 GPU au lieu de MobileViT-XXS CPU). Conservé pour la matrice de confusion par classe disponible dans `results/test_results.json`.

## 4. Verdict

- **Binaire (courant)** : modèle utilisable en pilote. 81 % val / 81 % test, F1 macro 80 %, AUC 88.5 %. À déployer en PyTorch Mobile Lite (`.ptl`), pas en TFLite (chaîne cassée) ni en TFJS (architecture incompatible).
- **Multi-classes (courant)** : pas encore entraîné. Combler les 3 classes manquantes et lancer le run complet avant de communiquer un chiffre.
- **Stage 2 fine-tune** : à essayer pour passer le plafond ~81 % du binaire — dégeler les 2-3 derniers blocs de l'attention, LR 1e-5, 3-5 epochs supplémentaires.
- **Calibrage de seuil** : avec AUC 88.5 % et un déséquilibre val 43/57, le seuil 0.5 standard sous-prédit `healthy`. Calibrer sur le split val (Youden's J) avant déploiement.
- **Test set externe** : aucun des runs n'a été évalué sur des photos hors-Food-101 (selfies, photos cuisine maison, traffic réel Whispr). C'est le risque numéro 1 avant un livrable produit.
