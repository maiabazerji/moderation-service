# Hyperparameter Sweep Report Generator

Outil de génération de rapport pour les balayages d'hyperparamètres, basé sur le pipeline d'entraînement EfficientNet en deux étapes. Exécute plusieurs configurations d'hyperparamètres et produit un rapport HTML comparatif.

## Prérequis

- Python 3.10+
- TensorFlow 2.x (environnement GPU recommandé)
- Jeu de données préparé (`train/dataset_merged/` contenant les sous-dossiers `Train/`, `Val/`, `Test/`)
- Dépendances : `numpy`, `matplotlib`, `scikit-learn`, `pyyaml`

## Démarrage rapide

```bash
cd src/efficientnet_lite_gpu

# 1. Lancer les 8 expériences
python -m tools.generate_training_report

# 2. Une fois l'entraînement terminé, regénérer uniquement le rapport
python -m tools.generate_training_report --report
```

## Arguments en ligne de commande

| Argument | Description |
|------|------|
| (aucun) | Lance toutes les expériences puis génère le rapport |
| `--report` | Saute l'entraînement, génère le rapport à partir des résultats existants |
| `--force` | Force la relance des expériences déjà terminées (sinon elles sont sautées) |
| `--experiments A,B,E` | Exécute uniquement les expériences dont l'ID est listé (séparés par des virgules) |

## Configurations d'expériences

Le script définit 8 configurations d'hyperparamètres :

| ID | Nom | Batch Size | S1 LR | S2 LR | S1 Epochs | S2 Epochs | Modèle | Description |
|----|------|-----------|-------|-------|-----------|-----------|-------|------|
| A | Baseline | 8 | 1e-3 | 2e-5 | 12 | 6 | B0 | Configuration de référence |
| B | Large Batch | 32 | 1e-3 | 2e-5 | 12 | 6 | B0 | Batch size élevé |
| C | High LR | 8 | 1e-2 | 1e-4 | 12 | 6 | B0 | Taux d'apprentissage élevé |
| D | Low LR + Long | 8 | 5e-4 | 1e-5 | 20 | 8 | B0 | Taux faible + entraînement long |
| E | EfficientNet-B1 | 8 | 1e-3 | 2e-5 | 12 | 6 | B1 | Modèle plus grand |
| F | No Fine-tune | 8 | 1e-3 | 0 | 20 | 0 | B0 | Stage 1 uniquement |
| G | Batch 16 + Mid LR | 16 | 2e-3 | 5e-5 | 15 | 6 | B0 | Configuration intermédiaire |
| H | Aggressive Fine-tune | 8 | 1e-3 | 1e-4 | 10 | 12 | B0 | Fine-tuning prolongé |

### Expériences personnalisées

Éditer la liste `EXPERIMENTS` dans `tools/generate_training_report.py`. Chaque expérience contient :

```python
{
    "id": "X",                    # identifiant unique (sert au nommage du dossier)
    "name": "X: My Experiment",   # nom affiché
    "overrides": {                # surcharge des champs de train_config
        "batch_size": 8,
        "stage1_learning_rate": 1e-3,
        "stage2_learning_rate": 2e-5,
        "stage1_epochs": 12,
        "stage2_epochs": 6,
        "fine_tune": True,
    },
    "model_overrides": {          # surcharge des champs de model_config
        "model_name": "efficientnet-b0",
    },
}
```

## Fichiers de sortie

Les résultats d'entraînement sont écrits dans `train/results_sweep/` :

```
train/results_sweep/
├── exp_A/                              # un dossier par expérience
│   ├── data_exploration/
│   │   ├── class_distribution.png
│   │   ├── dataset_statistics.png
│   │   └── sample_images.png
│   ├── evaluation_results/
│   │   ├── test_metrics.json           # métriques sur le jeu de test
│   │   ├── test_class_report.json      # métriques par classe
│   │   ├── test_confusion_matrix.npy   # matrice de confusion
│   │   ├── confusion_matrix.png
│   │   └── class_performance.png
│   ├── training_logs/
│   │   ├── training_history.json       # courbes d'entraînement
│   │   └── best_metrics.json           # meilleures métriques
│   └── training_results/
│       ├── training_config.json        # configuration d'entraînement
│       └── training_history.png
├── exp_B/
│   └── ...
├── hyperparameter_sweep_report.html    # rapport HTML final
└── hyperparameter_sweep_metrics.json   # métriques agrégées
```

## Contenu du rapport

Le rapport HTML comprend 6 sections (titres tels qu'ils apparaissent dans le rapport) :

1. **Dataset Statistics** — distribution des classes, répartition Train/Val/Test, cartes statistiques
2. **Hyperparameter Comparison** — comparaison des accuracy et F1, nuage Precision-Recall, table des paramètres
3. **Per-class F1-score Comparison** — histogramme groupé du F1 par classe pour toutes les configurations
4. **Metrics Radar** — radar multi-axes Accuracy / Precision / Recall / F1(W) / F1(M)
5. **Per-Configuration Details** — pour chaque config : courbes d'entraînement (Stage 1/Stage 2), heatmap de matrice de confusion, table des métriques par classe
6. **Final Summary** — tableau récapitulatif de toutes les configurations, meilleure config mise en évidence

Le rapport est un fichier HTML autonome (graphiques encodés en base64), ouvrable directement dans un navigateur.

## Reprise après interruption

Le script détecte les expériences déjà terminées (via les fichiers JSON clés dans le dossier de résultats) et les saute par défaut. Si l'entraînement est interrompu, relancer la commande reprend au point où elle s'est arrêtée :

```bash
# Saute automatiquement les expériences terminées, relance les autres
python -m tools.generate_training_report

# Si les résultats d'une expérience posent problème, la forcer à retourner
python -m tools.generate_training_report --experiments C --force
```

## Conseils d'utilisation

- Exécuter de préférence sur GPU — sur CPU chaque expérience peut prendre 30 à 60+ minutes
- Lancer d'abord 2–3 configurations clés (par ex. `--experiments A,E,F`) pour valider le flux avant la campagne complète
- Une fois l'entraînement terminé, `--report` permet d'itérer sur la mise en forme du rapport sans retraîner
