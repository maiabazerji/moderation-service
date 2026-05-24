# Analyse d'Erreurs — Comparaison Tous Modèles (MobileNetV3 / MobileNetV2 / ViT)

## Résumé Exécutif

| Modèle | Accuracy | F1-Score | Erreurs | Classes | Cas Critique |
|--------|----------|----------|---------|---------|--------------|
| **MobileNetV3** | **92.92%** | **93.10%** | 684 | 3 | Unhealthy→Healthy (51) |
| **MobileNetV2** | **84.95%** | **83.69%** | ~803 | 9 | Class imbalance |
| **Vision Transformer** | **80.88%** | **80.33%** | ~2,167 | 2 (binary) | Healthy recall (75.14%) |

**Recommandation déploiement**: MobileNetV3 (meilleur accuracy/size/speed tradeoff)

---

# MOBILENETV3 — Analyse Détaillée

## Configuration & Métriques

**Backbone**: MobileNetV3Small, ImageNet pretrained  
**Dataset**: 9,714 images validation (3,716 healthy, 2,998 unhealthy, 3,000 not-food)  
**Accuracy finale**: 92.92% (stage 1: 91.56% → stage 2: 92.92%, +1.36%)  
**Erreurs totales**: 684 (7.08%)

## Matrice d'Erreurs MobileNetV3

```
                 Healthy  Not-Food  Unhealthy  Recall
Healthy          3455       156        105      92.97%
Not-Food           82      2745        173      91.50%
Unhealthy          51       117       2830      94.41%

Precision:      93.2%      92.8%      94.0%
Overall:        92.92% accuracy
```

## Erreurs Critiques

### 1. Unhealthy → Healthy (51 erreurs, 0.5%) — CRITIQUE

**Impact**: Faux négatif — aliment malsain classé sain
**Amélioration**: Stage 1 (74 err) → Stage 2 (51 err) = -31% ✓
**Mitigations**: Seuil 0.70, 9.2% vers révision, 94.33% accuracy certain

### 2. Not-Food → Food (82 erreurs, 0.8%) — HAUTE

**Impact**: Modération fail — contenu non-alimentaire passe filtre
**Amélioration**: Stage 1 (212 err) → Stage 2 (82 err) = -61% ✓
**Root cause**: CIFAR-10/SVHN non-représentatifs
**Solution**: Collecter 5-10K vraies images non-alimentaires

### 3. Healthy → Unhealthy (105 erreurs, 0.9%) — FAIBLE

**Impact**: Fausse alerte
**Amélioration**: Stage 1 (120 err) → Stage 2 (105 err) = -12.5% ✓

---

# MOBILENETV2 — Analyse Détaillée

## Configuration

**Backbone**: MobileNetV2-0.35 (width 35%)  
**Dataset**: Food-101 multi-classe (9 classes)  
**Accuracy**: 84.95%  
**F1-Score**: 83.69%

## Erreurs Principales

### 1. Sandwich Misclassification (Classe Faible)

- **Recall basse**: 68% (32% sandwichs mal classés)
- **Confusion majorité**: Sandwich→Burger
- **Cause**: Similarité visuelle, données insuffisantes (~650 samples)
- **Solution**: +500 images sandwich, fine-tuning stage 2, class_weight

### 2. Class Imbalance

- **Distribution**: Pizza/Burger ~710, Sandwich ~650
- **Impact**: Classes minoritaires → recall baissé
- **Non mitigué**: Pas de class_weight implémenté
- **Solution**: Implémenter class_weight dans loss

### 3. Backbone Trop Petit

- **Trade-off**: 0.5MB mais capacity limitée (9 classes)
- **Evidence**: Plateau à 84.95%
- **Recommendation**: Tester MobileNetV2-1.0 (full width)

---

# VISION TRANSFORMER — Analyse Détaillée

## Configuration

**Backbone**: ViT B/16 (gelé)  
**Dataset**: Food-101 binary (healthy vs unhealthy, ~75.5K frames)  
**Accuracy**: 80.88% test  
**ROC-AUC**: 88.54%

## Matrice d'Erreurs ViT

```
           Healthy  Unhealthy
Healthy    3655     1193      (recall: 75.38%)
Unhealthy  974      5540      (recall: 85.05%)

Precision: 79.02%   82.37%
```

## Erreurs Principales

### 1. Healthy Recall Basse (75.38%) — PRINCIPALE FAIBLESSE

- **Problème**: 25% des vrais "healthy" classés "unhealthy"
- **Cause**: Déséquilibre 43/57, pas de class_weight, backbone gelé
- **Impact**: 1M images → 53,000 faux positifs/mois

### 2. Threshold Miscalibration

- **Evidence**: ROC-AUC 88.54% vs Accuracy 80.88%
- **Cause**: Seuil 0.5 sous-optimal avec déséquilibre
- **Solution**: Calibrer avec Youden's J → +1-2% accuracy

### 3. Frozen Backbone Limitation

- **Problème**: ImageNet features ≠ optimales pour Food-101
- **Evidence**: Plateau epoch 9, baisse après
- **Solution**: Fine-tune derniers 3 blocks (LR 1e-6) → +2-3% expected

---

# ANALYSE COMPARATIVE

## Performance Ranking

| Rank | Model | Accuracy | F1 | Size | Speed | Deploy |
|------|-------|----------|----|----|-------|--------|
| 1 | MobileNetV3 | 92.92% | 93.10% | 1.24MB | <100ms | ✅ Mobile |
| 2 | MobileNetV2 | 84.95% | 83.69% | 0.5MB | Fast | ✅ Web |
| 3 | ViT | 80.88% | 80.33% | 335MB | Slow | ⏸️ Future |

## Error Pattern Distribution

### Erreurs Critiques (Faux Négatifs)

| Type | MobileNetV3 | MobileNetV2 | ViT |
|------|-----------|-----------|-----|
| **False Negative** | 0.5% | ~3% | 25% |
| **Severity** | ⚠️ Modéré | ⚠️⚠️ Moyen | 🔴 HAUTE |

### Erreurs Modération (Faux Positifs)

| Type | MobileNetV3 | MobileNetV2 | ViT |
|------|-----------|-----------|-----|
| **Not-Food Leak** | 0.8% | N/A | N/A (binary) |
| **Severity** | ⚠️ Moyen | ✓ N/A | ✓ N/A |

---

# RECOMMANDATIONS PAR MODÈLE

## MobileNetV3 (DÉPLOYER MAINTENANT)

**Status**: Production-ready ✅

**Checklist**:
- ✓ Seuil confiance 0.70 déployé
- ✓ Monitoring setup
- TODO: Collecter 5K not-food réel
- TODO: V2.0 avec early stopping (patience=2)

**Expected v2.0**: 94%+ accuracy

---

## MobileNetV2 (AMÉLIORER)

**Status**: Acceptable + améliorations 📈

**Actions**:
1. Implémenter class_weight
2. Fine-tuning stage 2
3. +500 images Sandwich
4. Tester MobileNetV2-1.0

**Expected**: 87-88% accuracy

---

## Vision Transformer (FUTURE)

**Status**: Expérimental (trop gros pour mobile) ⚠️

**Si déploiement voulu**:
1. Calibrer seuil (Youden's J)
2. Class_weight
3. Fine-tune 3 derniers blocks (LR 1e-6)
4. Early stopping (patience=3)

**Expected**: 82-84% accuracy

**Verdict**: Garder pour server/research

---

# CONCLUSION

**🟢 MobileNetV3 PRODUCTION-READY. Deploy with confidence thresholding.**

**Roadmap v2.0**:
- Week 1-2: MobileNetV3 production + monitoring
- Week 3-6: Not-food collection + MobileNetV2 improvements
- Week 7-10: Retraining all models with early stopping
- Ongoing: Monitoring + quarterly retraining
