# Guide de déploiement de l'environnement — GPU AMD (ROCm)

> **Cible matérielle** : **serveur équipé d'une carte AMD** (environnement d'entraînement / inférence de production).
> Pour le laptop de développement équipé d'un GPU NVIDIA, se référer à [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md).

**Projet** : moderation-service / `src/efficientnet_lite_gpu/`
**Framework** : TensorFlow 2.x (Keras) via ROCm
**Ticket associé** : WHISPR-889

---

## 1. Contexte

Ce document décrit la mise en place d'un environnement GPU **AMD + ROCm** sur le **serveur de production / d'entraînement long**. L'objectif est de pouvoir exécuter le même pipeline `src/efficientnet_lite_gpu/` que sur le laptop NVIDIA, sans modification du code applicatif — TensorFlow-ROCm expose la même API que TensorFlow-CUDA.

> ⚠️ **Note importante sur l'hétérogénéité de l'équipe** :
> - **Laptop de développement** → GPU **NVIDIA** (voir `SETUP_CUDA_NVIDIA.md`, CUDA)
> - **Serveur d'entraînement / inférence** → GPU **AMD** (ce document, ROCm)
>
> Avant toute montée en production, le pipeline doit être validé **sur les deux cibles**.

---

## 2. Pré-requis matériels et système

| Composant | Exigence générique | Configuration réelle du serveur |
|---|---|---|
| GPU | AMD Instinct / Radeon PRO / RDNA2+ supporté par ROCm | **2× AMD Radeon RX 6600 XT** (Navi 23, `gfx1030`) |
| VRAM | ≥ 8 GB | 8 GB par GPU |
| OS | Ubuntu 22.04 / RHEL 9 officiellement | **Debian 12 (bookworm)** — support communautaire |
| Kernel | ≥ 5.15 | 6.1.0-21-amd64 |
| CPU | x86_64 moderne | Intel Xeon E5-2699 v3 (18C/36T @ 2.30 GHz) |
| RAM | ≥ 32 GB recommandés | **15 GB** (contrainte à surveiller pour les gros sweeps) |
| Stockage | ≥ 100 GB (datasets + checkpoints) | 456 GB / (87 % utilisé, 57 GB libres à avril 2026) |

> ⚠️ ROCm **ne supporte pas** Windows pour l'entraînement TensorFlow. Le serveur doit être sous Linux natif (pas de WSL2).

> ⚠️ **RX 6600 XT (gfx1030) non officiellement listé** dans la matrice ROCm AMD. Il fonctionne avec la variable d'environnement `HSA_OVERRIDE_GFX_VERSION=10.3.0` (voir §5.1 et §7). En production il serait préférable de migrer vers un GPU officiellement supporté (MI210, W7800) pour bénéficier du support AMD complet.

> ⚠️ **RAM limitée à 15 GB** : pour les sweeps `results_sweep/` (jusqu'à 7 expériences avec chacun `dataset_raw_*/` de ~2 GB montés en mémoire via cache TF), surveiller `free -h` pendant les runs. Possible OOM système à craindre avant un OOM VRAM.

---

## 3. Matrice de compatibilité TensorFlow / ROCm / AMD

| TensorFlow (stock) | ROCm | Python | OS officiels |
|---|---|---|---|
| 2.19.x | 6.x / 7.x | 3.9 – 3.12 | Ubuntu 22.04 / 24.04 |
| 2.16.x – 2.18.x | 6.x | 3.9 – 3.12 | Ubuntu 22.04 |
| 2.14.x – 2.15.x | 5.7 (via `tensorflow-rocm`) | 3.9 – 3.11 | Ubuntu 22.04 |
| 2.13.x | 5.6 (via `tensorflow-rocm`) | 3.9 – 3.11 | Ubuntu 20.04 / 22.04 |

**Configuration réelle validée sur le serveur** : **TensorFlow 2.19 (stock) + ROCm 7.2.1 + Python 3.11.2** sur **Debian 12**, avec 2× RX 6600 XT (`gfx1030`) activées via `HSA_OVERRIDE_GFX_VERSION=10.3.0`.

> ℹ️ **Changement de paradigme depuis ROCm 6** : le paquet historique `tensorflow-rocm` tend à être déprécié. À partir de ROCm 6+, AMD recommande d'utiliser le paquet **stock `tensorflow`** (déjà compatible avec le runtime ROCm installé au niveau système). Sur le serveur actuel, `pip show tensorflow` renvoie `tensorflow 2.19.0` et la détection GPU fonctionne directement.

### 3.1 GPUs AMD officiellement supportés (ROCm 7.x)

- Instinct MI210 / MI250 / MI300 (serveur pro — pleinement supportés)
- Radeon PRO W6800 / W7800 / W7900
- Radeon RX 7900 XT / XTX
- Radeon RX 6800 / 6900 XT — support communautaire via `HSA_OVERRIDE_GFX_VERSION`
- **RX 6600 / 6600 XT** (notre serveur) — support communautaire via `HSA_OVERRIDE_GFX_VERSION=10.3.0`

Liste officielle : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>

---

## 4. Installation ROCm

### 4.1 Préparation système

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget gnupg2 curl lsb-release
```

### 4.2 Installation ROCm 7.x (méthode utilisée sur le serveur actuel)

Sur **Debian 12** (cas non officiel mais fonctionnel) ou Ubuntu 22.04/24.04, utiliser l'installeur `amdgpu-install` en le forçant sur la cible Ubuntu équivalente :

```bash
# Ubuntu 22.04 (officiellement supporté)
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/jammy/amdgpu-install_7.2.60201-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.60201-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

> Sur **Debian 12** (configuration réelle du serveur) : installer le paquet `.deb` Ubuntu jammy fonctionne en général grâce à la compatibilité binaire, mais certaines dépendances (`lsb-release`, `libtinfo5`) peuvent demander un lien symbolique ou `apt --fix-broken install`. Vérifier la version installée avec `cat /opt/rocm/.info/version` (doit afficher `7.2.1`).

Le flag `--no-dkms` est recommandé quand on utilise le pilote `amdgpu` déjà inclus dans le kernel (kernel ≥ 5.15). Sur un serveur headless, `--usecase=rocm` suffit.

### 4.4 Droits utilisateur

L'utilisateur qui lance l'entraînement doit appartenir aux groupes `render` et `video` :

```bash
sudo usermod -aG render,video $USER
# Se déconnecter / reconnecter
```

### 4.5 Redémarrage et vérification

```bash
sudo reboot

# Après reboot
rocminfo | head -30
# Doit afficher les GPUs détectés et leur gfx target (ex: gfx1030, gfx90a...)

rocm-smi
# Doit afficher la liste des GPUs, leur charge, température, VRAM
```

---

## 5. Environnement Python

Sur le serveur actuel (ROCm 7.2.1, Python 3.11.2) on utilise le **TensorFlow stock** directement :

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install --upgrade pip

# Depuis ROCm 6+ : tensorflow stock suffit, le runtime ROCm est pris depuis /opt/rocm
pip install -r requirements.txt
# tensorflow==2.19.0 (notre version serveur) fonctionne avec ROCm 7.x
```

> ℹ️ **Évolution par rapport à ROCm 5.x** : il n'y a plus besoin d'installer `tensorflow-rocm` en premier et de filtrer le `requirements.txt`. Le `tensorflow` stock (≥ 2.16) détecte automatiquement l'installation `/opt/rocm` et utilise HIP. C'est cette méthode qui est en place sur le serveur.

> ⚠️ Si vous êtes encore sur ROCm ≤ 5.7, utiliser à la place :
> ```bash
> pip install tensorflow-rocm==2.14.0
> grep -v '^tensorflow' requirements.txt | pip install -r /dev/stdin
> ```

### 5.1 Variables d'environnement utiles (OBLIGATOIRES sur RX 6600 XT)

Sur notre serveur équipé de **RX 6600 XT**, la variable `HSA_OVERRIDE_GFX_VERSION` est **indispensable** pour que ROCm accepte les GPUs non listés. Placer dans `~/.bashrc` ou dans un script `setup_rocm.sh` sourcé avant chaque run :

```bash
# RX 6600 XT = Navi 23 = gfx1030 → override vers 10.3.0 (RX 6800/6900 XT déjà supporté)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Limiter la visibilité GPU (équivalent de CUDA_VISIBLE_DEVICES)
# Notre serveur a 2 GPUs, utiliser seulement le premier :
export HIP_VISIBLE_DEVICES=0
# Ou les deux (mais data parallel TF nécessite une stratégie MirroredStrategy explicite) :
# export HIP_VISIBLE_DEVICES=0,1

# Log verbeux pour debug
# export AMD_LOG_LEVEL=3
```

---

## 6. Vérification de l'installation

### 6.1 Commandes système

```bash
rocminfo | grep -E "Name:|gfx"
rocm-smi --showproductname --showmeminfo vram
```

### 6.2 Test TensorFlow

```python
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
# Sortie attendue : [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

with tf.device('/GPU:0'):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    c = tf.matmul(a, b)
    print("Matmul ROCm OK — shape:", c.shape)
```

### 6.3 Adaptation du script `hardware_test.py`

Le script actuel `tools/hardware_test.py` utilise `tools_nvidia_cuda.py` et appelle `nvidia-smi`. Sur AMD il faut :

- Soit sauter les checks NVIDIA (le script affiche un warning inoffensif si `nvidia-smi` n'existe pas)
- Soit compléter le diagnostic avec `rocm-smi` (piste d'amélioration, ticket à ouvrir)

Pour l'instant, la détection TensorFlow (`tf.config.list_physical_devices('GPU')`) fonctionne de façon identique entre CUDA et ROCm.

---

## 7. Lancement du pipeline d'entraînement

Aucune modification de code n'est requise — TensorFlow-ROCm expose la même API.

```bash
cd src/efficientnet_lite_gpu
source .venv-rocm/bin/activate

# Si GPU non officiellement supporté, forcer le gfx target
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # adapter selon votre carte

python -m tools.hardware_test
python -m train.train
```

---

## 8. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `rocm-smi` vide ou erreur | Pilote `amdgpu` non chargé | `lsmod \| grep amdgpu`, `sudo modprobe amdgpu`, rebooter |
| `GPUs: []` dans TF | `tensorflow` (CUDA) écrasé sur `tensorflow-rocm` | `pip uninstall tensorflow tensorflow-rocm && pip install tensorflow-rocm==2.14.0` |
| `HSA Error: No GPU devices found` | GPU non reconnu par ROCm | Vérifier compatibilité gfx, tester `HSA_OVERRIDE_GFX_VERSION` |
| `Permission denied` sur `/dev/kfd` | Utilisateur pas dans `render` / `video` | `sudo usermod -aG render,video $USER` puis reconnexion |
| `OOM` sur batch 32 | VRAM insuffisante | Réduire `batch_size` dans `config.yaml`, ou activer `tf.config.experimental.set_memory_growth` |
| Entraînement lent / CPU only | ROCm non initialisé correctement | Vérifier `AMD_LOG_LEVEL=3` pour logs détaillés |

---

## 9. Docker (alternative recommandée pour la production)

AMD fournit des images officielles qui évitent toute la complexité de l'installation manuelle.

```bash
docker pull rocm/tensorflow:rocm5.7-tf2.14-dev

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size 16G \
  -v $(pwd):/workspace \
  rocm/tensorflow:rocm5.7-tf2.14-dev bash
```

À l'intérieur du conteneur :

```bash
cd /workspace/src/efficientnet_lite_gpu
pip install -r requirements.txt  # tensorflow-rocm est déjà fourni par l'image
python -m train.train
```

Catalogue complet : <https://hub.docker.com/r/rocm/tensorflow/tags>

---

## 10. Environnement de référence — serveur actuel

Configuration exacte mesurée sur le serveur d'entraînement (avril 2026) :

| Composant | Valeur |
|---|---|
| OS | **Debian 12 (bookworm)** |
| Kernel | 6.1.0-21-amd64 |
| CPU | Intel Xeon E5-2699 v3 (18C/36T @ 2.30 GHz) |
| RAM | 15 GB |
| GPU | **2× AMD Radeon RX 6600 XT** (Navi 23, `gfx1030`, 8 GB VRAM chacune) |
| Pilote | `amdgpu` (kernel stock) |
| **ROCm** | **7.2.1** |
| Python | 3.11.2 |
| TensorFlow | **2.19.0** (stock, pas `tensorflow-rocm`) |
| Variable obligatoire | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Stockage | 456 GB (87 % utilisé) |

### 10.1 Checks rapides pour valider l'environnement serveur

```bash
# Version ROCm installée
cat /opt/rocm/.info/version         # → 7.2.1

# GPUs vus par le runtime ROCm
rocminfo | grep -E "Name:.*gfx"     # → Name: gfx1030 (×2)
rocm-smi --showproductname           # → 2× RX 6600 XT

# TensorFlow
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
# → 2.19.0
# → [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), ...:GPU:1...]
```

---

## 11. Différences CUDA ↔ ROCm à retenir

| Aspect | NVIDIA / CUDA (laptop) | AMD / ROCm (serveur) |
|---|---|---|
| Paquet TF | `tensorflow` | `tensorflow-rocm` |
| Outil monitoring | `nvidia-smi` | `rocm-smi` + `rocminfo` |
| Variable visibilité | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` |
| OS Windows | Supporté (WSL2) | Non supporté |
| Docker image | `tensorflow/tensorflow:*-gpu` | `rocm/tensorflow:*` |
| Override arch | — | `HSA_OVERRIDE_GFX_VERSION` |

Le code applicatif du pipeline (`train/`, `tools/`) **ne doit pas** avoir de branchement spécifique NVIDIA / AMD : la différence est gérée au niveau du package TensorFlow installé.

---

## 12. Références

- Documentation ROCm : <https://rocm.docs.amd.com/>
- Guide d'installation ROCm Linux : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>
- TensorFlow-ROCm : <https://github.com/ROCm/tensorflow-upstream>
- Matrice de support GPU AMD : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>
- Images Docker ROCm : <https://hub.docker.com/r/rocm/tensorflow>
- Équivalent NVIDIA (laptop) : [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md)
