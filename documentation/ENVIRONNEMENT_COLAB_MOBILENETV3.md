# Environnement Google Colab — MobileNetV3Small

Notes sur l'environnement d'exécution attendu par `src/mobilenet_v3_small/mobilenet_colab.ipynb`. Couvre l'installation des dépendances, l'authentification HF / Drive, les chemins persistés et les pièges du runtime Colab.

## Pourquoi Colab et pas local

- **GPU gratuit** (T4 16 Go le plus souvent, parfois V100 / A100 si chance). Suffisant pour un MobileNetV3Small à 32×224×224 sans contrainte mémoire.
- **Drive monté** pour la persistance des checkpoints malgré les déconnexions de runtime (rythme typique : ~12 h max d'uptime, déconnexion après ~90 min d'inactivité).
- **Datasets HF** téléchargés directement avec `hf_transfer` (×3-5 plus rapide qu'en local résidentiel).

L'inconvénient connu : le runtime peut tomber à n'importe quel moment. Le notebook est écrit pour être **résilient à la déconnexion** — chaque cellule rétablit son état si elle est ré-exécutée seule.

## Pile logicielle

Cellule 1 du notebook :

```python
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                "tensorflow", "tensorflowjs", "tensorflow-decision-forests"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tensorflow==2.19.0",
                "tensorflow-decision-forests==1.12.0",
                "tensorflowjs",
                "huggingface_hub",
                "hf_transfer"])
```

| Paquet | Version | Pourquoi épinglée |
|---|---|---|
| `tensorflow` | **2.19.0** | Évite l'erreur `undefined symbol` rencontrée avec TF par défaut de l'image Colab |
| `tensorflow-decision-forests` | **1.12.0** | Compatible avec TF 2.19.0 (la version par défaut de Colab pointe sur un TF différent et plante au load) |
| `tensorflowjs` | non épinglée | Laisse pip résoudre une version compatible TF 2.19.0 |
| `huggingface_hub` | non épinglée | API HF (download / upload) |
| `hf_transfer` | non épinglée | Backend Rust pour accélérer les transferts HF |

Activation explicite :

```python
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

Versions observées en sortie de notebook (cellule 8) :

- `torch == 2.10.0+cu128`
- `transformers == 5.0.0`

Torch / Transformers ne sont pas utilisés par le pipeline d'entraînement, mais sont préinstallés dans l'image Colab et peuvent servir aux exports / validations annexes.

## Authentification

### Hugging Face

```python
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ⚠ token en dur dans la cellule 1
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN, add_to_git_credential=False)
```

**Recommandation** : déplacer le token dans les **Colab Secrets** (icône clé dans la barre latérale gauche) et lire :

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
```

Le token codé en dur dans le notebook actuel a déjà déclenché un `401 Unauthorized` sur l'upload de la cellule 5 — probablement révoqué. Tout commit public du notebook expose le token.

### Google Drive

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
DRIVE_DIR = Path("/content/drive/MyDrive/whispr-checkpoints/mobilenet")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
```

Le `mount` ouvre une popup OAuth la première fois (à valider manuellement dans le navigateur). Une fois autorisé, le token reste valide tant que la session Colab vit.

Si l'utilisateur **refuse** ou ferme la popup : `DRIVE_DIR = None` et le notebook continue sans persistance — les checkpoints sont en local uniquement et **perdus** à la déconnexion. Le notebook l'imprime mais ne plante pas.

## Chemins persistés

| Chemin | Contenu | Persistance |
|---|---|---|
| `/content/frames/` | Images brutes téléchargées (Food-101 + Fashion/CIFAR/SVHN) | Local (volatile) |
| `/content/binary/` | Symlinks par classe (`healthy`, `unhealthy`, `not_food`) | Local (volatile, recréé à chaque run) |
| `/content/model.keras` | Checkpoint actif | Local + copié sur Drive à chaque epoch |
| `/content/train_state.json` | Epoch courante + historique | Local + Drive |
| `/content/model.tflite`, `model_quantized.tflite` | Exports TFLite | Local |
| `/content/model_tfjs/` | Export TFJS (`model.json` + shards `.bin`) | Local + uploadé HF |
| `/content/confusion_matrix.png` | Visualisation post-entraînement | Local |
| `/content/drive/MyDrive/whispr-checkpoints/mobilenet/` | Copie Drive des checkpoints | **Persistante** |

Le seul vrai mécanisme de survie inter-runtime est `DRIVE_DIR`. Si on perd la session, on remonte Drive sur un nouveau runtime et la cellule 4 voit `DRIVE_CKPT` / `DRIVE_STATE` et restaure automatiquement.

## Comportement face aux déconnexions

Le notebook est conçu cellule par cellule pour être ré-exécutable sans repartir de zéro :

| Cellule | Action en cas de re-run | Note |
|---|---|---|
| 1 (setup) | Réinstalle les paquets, reconnecte HF + Drive | Idempotent |
| 2 (dataset) | Compte `frames/*.jpg`. Si ≥ 1000, skip fallback | Évite re-télécharger Food-101 |
| 3 (binary/) | `rmtree` + recrée les symlinks | Rapide, OK |
| 4 (train) | Charge `model.keras` + `train_state.json`, reprend à `last_epoch` | Cœur de la résilience |
| 5 (HF upload) | Échoue silencieusement si token invalide | Pas bloquant |
| 6/7 (exports) | Re-télécharge depuis HF si modèle local absent | Fallback propre |

Garde-fou en cellule 4 :

```python
FORCE_RETRAIN = True  # @param
```

Quand activé, supprime `model.keras` / `train_state.json` (local + Drive) avant de démarrer. Pratique pour relancer un entraînement propre, **destructif** pour un entraînement long en cours. Le passer à `False` avant un run de référence.

## GPU disponible

À vérifier en début de session :

```python
!nvidia-smi
```

Repères :
- **T4** (le plus courant) : ~85 s / epoch sur ce notebook (44 800 images / 32 batch).
- **V100** / **A100** : 50-60 s / epoch — pas garanti, dépend de la disponibilité Colab.
- **CPU only** (cas où Colab refuse le GPU) : ~25-30 min / epoch — non viable, redémarrer le runtime jusqu'à obtenir un GPU.

Forcer le sélecteur Colab : *Runtime → Change runtime type → Hardware accelerator → GPU* avant de lancer la cellule 1.

## Limites du runtime Colab

- **Quotas** : un compte Colab gratuit ouvert toute la journée finit par tomber sur des refus "no GPU available" — basculer sur un autre compte ou attendre.
- **Disconnect après inactivité** : ~90 min sans interaction navigateur ⇒ runtime coupé. Le snippet JS classique (forcer un click périodique dans la console DevTools) contourne mais reste fragile.
- **Espace disque /content** : ~80 Go alloués mais partagés entre `/tmp`, datasets, parquets HF, exports. Le dataset complet (56 000 images JPEG q88) tient sous 1 Go, marge confortable.
- **RAM** : ~12 Go en édition gratuite. La cellule 4 monte à ~8-9 Go pic pendant le chargement du backbone ImageNet. Pas de problème observé.

## Checklist avant de lancer un run propre

1. Token HF valide et placé dans les Colab Secrets (pas en dur dans le code).
2. Drive monté et `whispr-checkpoints/mobilenet/` vide (ou contenant le checkpoint à reprendre).
3. GPU confirmé via `!nvidia-smi`.
4. `FORCE_RETRAIN = False` si on reprend, `True` si on repart de zéro.
5. `TOTAL_EPOCHS` ajusté (10 suffisent dans la config gelée actuelle — voir `RAPPORT_ENTRAINEMENT_MOBILENETV3_FR.md`).
6. Cellule 5 (upload HF) : vérifier que `HF_MODEL` pointe vers le bon dépôt (`maia2000/mobilenet-food-binary` pour le run de référence).
