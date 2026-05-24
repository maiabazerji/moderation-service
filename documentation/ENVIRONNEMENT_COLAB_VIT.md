# Environnement Google Colab — ViT Video Classifier

Notes sur l'environnement d'exécution attendu par `src/vit_video/vit_video.ipynb` (multi-classes) et `vit_video_binary.ipynb` (binaire). Couvre le clone repo, l'install PyTorch/timm, l'authentification HF / Drive, et les pièges spécifiques au modèle ViT sous Colab.

## Pourquoi Colab et pas local

Trois raisons principales :

- **GPU CUDA gratuit** : ViT-B/16 a ~86 M paramètres, l'entraînement en CPU descend à `mobilevit_xxs` (~1.3 M params) pour rester praticable. Le notebook auto-détecte le GPU et bascule sur ViT-B/16 si CUDA est disponible.
- **Drive monté** pour la persistance du checkpoint (`best_food_classifier.pth`) — sans Drive, une déconnexion du runtime tue l'entraînement.
- **`yt-dlp` + `ffmpeg`** déjà disponibles dans l'image Colab pour le téléchargement YouTube (mode A du dataset, voir `DATASET_PREPARATION_VIT.md`).

## Pile logicielle

### Clone du repo (cellule 1)

```python
REPO_URL = "https://github.com/whispr-messenger/moderation-service.git"
BRANCH   = "WHISPR-668"
```

Clone shallow (`--depth 1`) sur la branche `WHISPR-668`, fallback sur `main` si la branche n'existe plus. Le notebook fait :

```python
SRC_DIR = "/content/moderation-service/src"
sys.path.insert(0, SRC_DIR)
```

…pour rendre `vit_video.*` importable, puis `os.chdir(VIT_DIR)` pour que les chemins relatifs marchent. Les modules `vit_video` déjà chargés en cache sont purgés (`del sys.modules[...]`) — utile en cas de re-clone après modification.

### Dépendances (cellule install)

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "icrawler", "huggingface_hub"], check=True)
```

Le `requirements.txt` du module contient (voir `src/vit_video/requirements.txt`) :

| Paquet | Rôle |
|---|---|
| `torch`, `torchvision` | Backbone ViT (torchvision) + training loop |
| `timm` | Backbones MobileViT, fallback CPU |
| `opencv-python` | Extraction frames + inference webcam |
| `yt-dlp`, `imageio-ffmpeg` | Téléchargement YouTube + ffmpeg portable |
| `scikit-learn` | Métriques + k-fold CV (`validate_model.py`) |
| `huggingface-hub` | Download dataset + upload modèle |
| `onnx`, `onnxruntime` | Export ONNX (optionnel) |
| `ai-edge-torch` | Export TFLite (optionnel, recommandé Android) |
| `coremltools` | Export CoreML (macOS uniquement) |
| `icrawler` | Scraping Bing pour la classe `not_food` (notebook only) |

Versions observées (cellule env-check) : `torch >= 2.1`, `torchvision`, `timm >= 0.9`. Pas de pin strict — le notebook accepte ce qui arrive dans l'image Colab.

### GPU CUDA (cellule env-check)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Helper interne :

```python
from vit_video.utils.hardware import get_device, print_device_info
device = get_device()
```

`get_device()` retourne `cuda`, `mps` (macOS Apple Silicon) ou `cpu` selon la disponibilité. La sélection du backbone (`vit_b_16` vs `mobilevit_xxs`) suit la même logique côté `train.py`.

## Authentification

### Hugging Face (cellule 3)

Cascade de résolution du token, dans l'ordre :

1. **Colab Secrets** (`from google.colab import userdata; userdata.get("HF_TOKEN")`) — recommandé.
2. **Variable d'environnement** `HF_TOKEN`.
3. **Cache fichier** `/content/.hf_token` (chmod 600).
4. **Prompt interactif** (`getpass`) — pour les sessions ad-hoc.

```python
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN, add_to_git_credential=False)
```

Sans token, le téléchargement public du dataset HF marche toujours, mais les **uploads sont sautés** (`SKIP_DATASET_UPLOAD` et `SKIP_MODEL_UPLOAD` se déclenchent).

### Google Drive (cellule drive)

```python
from google.colab import drive
drive.mount("/content/drive")
DRIVE_CHECKPOINT_DIR  = "/content/drive/MyDrive/whispr-checkpoints"
DRIVE_CHECKPOINT_PATH = f"{DRIVE_CHECKPOINT_DIR}/best_food_classifier.pth"
```

À la différence du notebook MobileNetV3, le `mount` n'est pas optionnel : sans Drive, on perd le checkpoint à chaque déconnexion. Le notebook tolère pourtant l'absence (`DRIVE_CHECKPOINT_PATH = None`, training continue sans sync), mais ce n'est pas recommandé pour un run multi-heures.

Reprise auto : la cellule de training vérifie `os.path.exists(DRIVE_CHECKPOINT_PATH)` et copie le checkpoint vers `models/best_food_classifier.pth` avant de lancer `train_main(...)` avec `resume_from=...`.

## Chemins persistés

| Chemin | Contenu | Persistance |
|---|---|---|
| `/content/moderation-service/` | Repo cloné | Local (volatile) |
| `/content/moderation-service/src/vit_video/food_data/raw_videos/` | MP4 téléchargés | Local (volatile, gros !) |
| `food_data/frames/<classe>/*_frame_*.jpg` | Frames pour le training | Local + uploadable HF |
| `food_data/video_split_manifest.json` | Splits train/val/test (vidéo-level) | Local + uploadable HF |
| `src/vit_video/models/best_food_classifier.pth` | Checkpoint actif | Local + Drive |
| `src/vit_video/models/best_food_classifier_history.json` | Loss/acc par epoch | Local |
| `src/vit_video/exported_models/*.{pth,pt,ptl,onnx,tflite}` | Exports mobile | Local (uploadable HF) |
| `src/vit_video/results/test_results.json` | Métriques test + matrice confusion | Local |
| `/content/drive/MyDrive/whispr-checkpoints/best_food_classifier.pth` | Copie Drive du checkpoint | **Persistante** |
| `/content/.hf_token` | Token HF cached chmod 600 | Local (session-life) |

Le couple Drive + token cache survit aux re-runs de cellules. Tout le reste est volatile et reconstruit par le notebook.

## Comportement face aux déconnexions

| Cellule | Action en cas de re-run | Note |
|---|---|---|
| 1 (clone) | Skip si `vit_video/__init__.py` présent | Idempotent |
| Drive | `force_remount=False`, ne re-mount pas si déjà actif | Idempotent |
| 3 (HF token) | Cascade Secrets → env → cache → prompt | Skip silencieux si rien |
| 5 (HF download) | Skip si dataset déjà sur disque | Cf. `frames_directory_has_images` |
| 5a (Food-101) | `if dst.exists(): continue` par image | Idempotent |
| **5b (Bing)** | **Commentée dans la version courante du notebook** | Décommenter pour réactiver |
| **6 (HF upload)** | **Commentée dans la version courante du notebook** | Dépôt déjà en place |
| 8 (split manifest) | `ensure_split_manifest` détecte un manifest existant et le réutilise | Manifest figé |
| 9 (train) | Resume depuis Drive si checkpoint présent | Cœur de la résilience |

Le notebook est donc **safe à re-lancer cellule par cellule** après une déconnexion — il faut juste re-clicker `Runtime → Run all`.

⚠ Les cellules 5b et 6 étant commentées, un run from-scratch (HF vide) sur le multi-classes produira un dataset incomplet (13 classes sur 16, voir `DATASET_PREPARATION_VIT.md` §B.6). Décommenter explicitement avant de lancer.

## GPU disponible

À vérifier en début de session :

```python
!nvidia-smi
```

Repères pour ce modèle :

| GPU | Backbone | Temps/epoch (estimé) |
|---|---|---|
| **T4** (free, ~16 Go) | ViT-B/16 + BiLSTM | 12-18 min |
| **V100** / **A100** | ViT-B/16 + BiLSTM | 6-10 min |
| **CPU only** | MobileViT-XXS + avg | 30-50 min (non viable au-delà de 5 epochs) |

Forcer le sélecteur : *Runtime → Change runtime type → Hardware accelerator → GPU* avant la cellule 1.

Si Colab ne donne pas de GPU (quota épuisé), le notebook **continue** mais bascule sur MobileViT-XXS sans le dire explicitement — c'est ce qui explique l'ancien run CLI (run CPU 4 epochs, voir `RAPPORT_ENTRAINEMENT_VIT.md`). Surveiller la sortie de `print_device_info(...)` à la cellule env-check.

## Limites du runtime Colab pour ce modèle

- **Mémoire GPU** : ViT-B/16 + batch 8 + 8 frames = ~14-15 Go en VRAM. Sur T4 (16 Go), ça passe mais à la limite. Si OOM, baisser `--batch-size 4` ou `--num-frames 4`.
- **Inactivity disconnect** : ~90 min sans interaction tue le runtime. Un entraînement multi-heures (16 classes, 20 epochs) **doit** persister son checkpoint sur Drive. Sinon, perte totale au moindre fermeture d'onglet.
- **Espace disque /content** : ~80 Go partagés. Le dataset vidéo (vidéos YouTube + frames) peut monter à 5-10 Go selon `--videos-per-keyword`. Si l'espace manque pendant le scraping, `yt-dlp` plante avec `OSError: No space left on device`.
- **Quotas Colab** : compte gratuit limité à ~12 h cumulées de GPU/jour. Au-delà, la session refuse le GPU jusqu'au lendemain. Colab Pro lève cette contrainte (~$10/mois).

## Différences avec le notebook binaire (`vit_video_binary (2).ipynb`)

Nouvelle version du notebook binaire (`vit_video_binary (2).ipynb`). **Beaucoup plus simple que le multi-classes** — pas de clone du repo, pas de Drive, dataset streamé directement depuis HF.

### Pile et auth

```python
subprocess.run(["pip","-q","install","huggingface_hub","hf_transfer",
                "datasets","scikit-learn"], check=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN").strip()
```

Strictement **les Colab Secrets** comme source de token, pas de cascade ni de prompt. Aucun clone de repo — tout le code tient dans le notebook (9 cellules au total).

### Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Backbone | ViT-B/16 (`vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`) |
| Backbone gelé | Oui — seule la tête `nn.Linear(768, 2)` est entraînée |
| Optimizer | AdamW `lr=1e-3, weight_decay=1e-4` |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |
| AMP | Activé |
| Batch size | 64 (CUDA) / 8 (CPU) |
| Epochs | 10 max, `EarlyStopping(patience=3)` sur `val_acc` |

### Dataset

Source unique : **Food-101** (`ethz/food101`), 75 750 images train, split 70/15/15 image-level (pas de notion vidéo). Mapping binaire en mémoire via un set `UNHEALTHY` de 58 classes (cf. `DATASET_PREPARATION_VIT.md` §Mode C).

Aucun frame n'est écrit sur disque — `HFImageDs` lit chaque exemple via `ds[i]["image"]` à la demande. Avantage : démarrage rapide (~3-4 min de téléchargement parquet la première fois, ensuite cache HF local). Inconvénient : un `Runtime → Disconnect` perd le cache `/content` et il faut re-télécharger les 5 Go de parquets.

### Exports et HF

Cellule 6 produit :

```
/content/model.torchscript.pt   →  335 MB
/content/model_mobile.ptl       →  335 MB  (PyTorch Mobile Lite Interpreter)
/content/model.onnx             →  101 KB  ⚠ anormalement petit
```

Note : le `model_mobile.ptl` (335 MB) est le format mobile canonique, mais **335 MB ce n'est PAS un livrable mobile sain**. Pour une vraie cible mobile, partir de MobileViT-XXS (notebook multi-classes) ou MobileNetV3 (autre pipeline). ViT-B/16 frozen est correct pour serveur ou app desktop, pas pour un APK/IPA.

Cellule 7 push vers **`maia2000/food-classifier-binary-vit`**. Cellule 8 (la dernière ajoutée) tente une conversion TFLite supplémentaire — **elle échoue** sur `Exporting a ScriptModule is not supported` (voir `RAPPORT_ENTRAINEMENT_VIT.md` §1.6).

### Wall-clock

~10 min sur T4 (10 epochs × ~50 s/epoch sur 53 k images × batch 64 + ~3 min de téléchargement parquet la première fois).

## Checklist avant un run propre

### Multi-classes (`vit_video.ipynb`)

1. Token HF valide dans les Colab Secrets (clé `HF_TOKEN`).
2. Drive monté et `whispr-checkpoints/` accessible.
3. GPU confirmé via `!nvidia-smi` **et** via `print_device_info(...)`.
4. Branche cohérente : `BRANCH = "WHISPR-668"` ou la branche courante.
5. **Décommenter les cellules 5b (Bing) et 6 (HF upload)** si on veut un dataset complet 16 classes — sinon, 3 classes resteront vides (`smoothies`, `sugary_drinks`, `not_food`).
6. Supprimer `food_data/frames/_tmp_download/` s'il existe (résidu de scrape).
7. `EXPORT_FORMATS` à jour : `['torchscript', 'onnx', 'tflite', 'tfjs']` est le défaut actuel. Pour TFJS, vérifier que le backbone résolu est MobileViT-XXS (ViT-B/16 produirait un bundle navigateur de ~330 MB inutilisable).
8. `HF_MODEL_REPO` pointe vers **`maia2000/food-classifier-vitb16`** (le nom a changé, plus `maia2000/food-classifier`).
9. Si reprise : vérifier que `DRIVE_CHECKPOINT_PATH` existe et a la bonne taille avant de lancer la cellule training.

### Binaire (`vit_video_binary (2).ipynb`)

1. Token HF dans les Colab Secrets — obligatoire (pas de fallback).
2. GPU confirmé (`device: cuda` dans la sortie de la cellule 4).
3. Espace disque `/content` ≥ 6 Go libres pour le cache Food-101.
4. Pas besoin de Drive — le run tient en ~10 min, pas de checkpoint à persister.
5. `HF_MODEL = "maia2000/food-classifier-binary-vit"` — vérifier si on doit écraser ou créer un nouveau dépôt avant de lancer la cellule 7.
6. Sauter la cellule 8 (TFLite conversion cassée) — le `.ptl` de la cellule 6 suffit pour la cible mobile.
