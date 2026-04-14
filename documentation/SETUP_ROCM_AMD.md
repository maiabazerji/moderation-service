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

| Composant | Exigence |
|---|---|
| GPU | AMD Radeon Instinct MI-series / Radeon PRO / RDNA2+ supporté par ROCm (voir section 3) |
| VRAM | ≥ 8 GB recommandés (batch=32 en 224×224) |
| OS | **Ubuntu 22.04 LTS** (principalement supporté) ou RHEL 9 |
| Kernel | ≥ 5.15 |
| RAM | ≥ 32 GB pour l'entraînement long |
| Stockage | ≥ 100 GB (datasets + checkpoints + sweeps) |

> ⚠️ ROCm **ne supporte pas** Windows pour l'entraînement TensorFlow. Le serveur doit être sous Linux natif (pas de WSL2).

---

## 3. Matrice de compatibilité TensorFlow-ROCm / ROCm / AMD

| tensorflow-rocm | ROCm | Python | OS |
|---|---|---|---|
| 2.14.x | 5.7 | 3.9 – 3.11 | Ubuntu 22.04 |
| 2.13.x | 5.6 | 3.9 – 3.11 | Ubuntu 20.04 / 22.04 |
| 2.12.x | 5.4 | 3.9 – 3.11 | Ubuntu 20.04 / 22.04 |

**Version recommandée pour ce projet** : `tensorflow-rocm==2.14.x` + ROCm 5.7.

### 3.1 GPUs AMD officiellement supportés (ROCm 5.7)

- Instinct MI210 / MI250 / MI300
- Radeon PRO W6800 / W7800 / W7900
- Radeon RX 7900 XT / XTX (support partiel — à valider)
- Radeon RX 6800 / 6900 XT (support communautaire via `HSA_OVERRIDE_GFX_VERSION`)

Liste officielle : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>

---

## 4. Installation ROCm sur Ubuntu 22.04

### 4.1 Préparation système

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget gnupg2 curl lsb-release
```

### 4.2 Ajout du dépôt ROCm

```bash
wget https://repo.radeon.com/amdgpu-install/5.7.1/ubuntu/jammy/amdgpu-install_5.7.50701-1_all.deb
sudo apt install -y ./amdgpu-install_5.7.50701-1_all.deb
```

### 4.3 Installation du stack ROCm

```bash
sudo amdgpu-install --usecase=rocm --no-dkms
```

> Le flag `--no-dkms` est recommandé si vous utilisez le pilote `amdgpu` déjà inclus dans le kernel. Sur un serveur sans écran (headless), préférer `--usecase=rocm` seul.

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

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install --upgrade pip

# TensorFlow-ROCm en priorité
pip install tensorflow-rocm==2.14.0

# Reste des dépendances SANS TensorFlow (pour éviter l'écrasement)
grep -v '^tensorflow' requirements.txt | pip install -r /dev/stdin
```

> ⚠️ **Piège classique** : `pip install -r requirements.txt` installerait `tensorflow` (version CUDA) par-dessus `tensorflow-rocm`. Toujours installer `tensorflow-rocm` **en premier** puis filtrer le requirements.

### 5.1 Variables d'environnement utiles

À placer dans `~/.bashrc` ou dans un script `setup_rocm.sh` :

```bash
# Forcer le GPU target pour les RDNA2/3 non officiellement supportés
# (exemple pour RX 6800 XT = gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Limiter la visibilité GPU (équivalent de CUDA_VISIBLE_DEVICES)
export HIP_VISIBLE_DEVICES=0

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

## 10. Environnement de référence cible

Configuration cible du serveur d'entraînement :

| Composant | Version |
|---|---|
| OS | Ubuntu 22.04 LTS (serveur, pas de bureau graphique) |
| GPU | AMD (serveur de production) |
| ROCm | 5.7.1 |
| Pilote | `amdgpu` kernel stock |
| Python | 3.11 |
| TensorFlow-ROCm | 2.14.0 |

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
