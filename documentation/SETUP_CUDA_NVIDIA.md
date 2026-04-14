# Guide de déploiement de l'environnement — GPU NVIDIA (CUDA)

> **Cible matérielle** : **ordinateur portable équipé d'une carte NVIDIA** (environnement de développement local).
> Pour le serveur de production équipé d'un GPU AMD, se référer à [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md).

**Projet** : moderation-service / `src/efficientnet_lite_gpu/`
**Framework** : TensorFlow 2.x (Keras)

---

## 1. Contexte

Ce document décrit comment préparer un environnement d'entraînement et d'inférence reproductible sur **laptop NVIDIA** (poste de développement). L'objectif est de pouvoir exécuter :

- Le pipeline d'entraînement `train/train.py` (MobileNetV2 / EfficientNet)
- Les outils de conversion `tools/convert_model.py` (Keras → TFLite / TFJS)
- Le sweep d'hyperparamètres et la génération des rapports

Le tout avec accélération GPU via CUDA + cuDNN.

> ⚠️ **Note importante sur l'hétérogénéité de l'équipe** :
> - **Laptop de développement** → GPU **NVIDIA** (ce document, CUDA)
> - **Serveur d'entraînement / inférence** → GPU **AMD** (voir `SETUP_ROCM_AMD.md`, ROCm)
>
> Toujours tester le pipeline sur les deux cibles avant de merger vers `main`.

---

## 2. Pré-requis matériels et système

| Composant | Exigence |
|---|---|
| GPU | NVIDIA avec Compute Capability ≥ 3.5 (RTX / GTX 10-series ou plus récent recommandé) |
| VRAM | ≥ 6 GB pour batch=32 en 224×224 (MobileNetV2) |
| OS | Ubuntu 20.04 / 22.04 LTS, Windows 10/11 (WSL2 recommandé sur Windows) |
| RAM | ≥ 16 GB |
| Stockage | ≥ 20 GB (datasets + checkpoints exclus) |

---

## 3. Matrice de compatibilité TensorFlow / CUDA / cuDNN

La compatibilité est **stricte** — une mauvaise combinaison fait planter `tf.config.list_physical_devices('GPU')` sans erreur claire.

| TensorFlow | Python | CUDA | cuDNN | Driver NVIDIA min. |
|---|---|---|---|---|
| 2.16.x | 3.9 – 3.11 | 12.3 | 8.9 | ≥ 545 |
| 2.15.x | 3.9 – 3.11 | 12.2 *(12.3 OK via forward-compat)* | 8.9 | ≥ 525 |
| 2.14.x | 3.9 – 3.11 | 11.8 | 8.7 | ≥ 450 |
| 2.13.x | 3.8 – 3.11 | 11.8 | 8.6 | ≥ 450 |

**Configuration réelle validée sur le laptop de dev** : **TensorFlow 2.15+ + CUDA 12.3 + cuDNN 8.9** sur une **RTX 3070 Laptop (8 GB)**.

Source officielle : <https://www.tensorflow.org/install/source#gpu>

---

## 4. Installation (Ubuntu 22.04)

### 4.1 Pilote NVIDIA

```bash
# Désinstaller tout pilote existant
sudo apt purge '^nvidia-.*' -y
sudo apt autoremove -y

# Installer le driver recommandé
sudo ubuntu-drivers autoinstall
sudo reboot
```

Vérification :

```bash
nvidia-smi
# Doit afficher le GPU, la version du driver (≥ 525) et la version CUDA max supportée
```

### 4.2 CUDA Toolkit 12.2

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run --toolkit --silent --override
```

Ajouter au `~/.bashrc` :

```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

Vérification :

```bash
nvcc --version
# Doit afficher « release 12.2 »
```

### 4.3 cuDNN 8.9

1. Télécharger **cuDNN 8.9 for CUDA 12.x** (compatible avec CUDA 12.3) depuis <https://developer.nvidia.com/cudnn> (compte NVIDIA requis)
2. Extraire et copier :

```bash
tar -xvf cudnn-linux-x86_64-8.9.*_cuda12-archive.tar.xz
sudo cp cudnn-*/include/cudnn*.h /usr/local/cuda-12.3/include/
sudo cp cudnn-*/lib/libcudnn* /usr/local/cuda-12.3/lib64/
sudo chmod a+r /usr/local/cuda-12.3/include/cudnn*.h /usr/local/cuda-12.3/lib64/libcudnn*
```

Vérification :

```bash
cat /usr/local/cuda-12.3/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 4.4 Environnement Python

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Installation (Windows 10/11 via WSL2 — recommandé)

### 5.1 Pré-requis côté Windows

1. Mettre à jour le pilote **NVIDIA GeForce pour Windows** (≥ 525, depuis GeForce Experience ou le site NVIDIA).
   **Ne pas installer de pilote Linux dans WSL2** : WSL2 utilise directement le pilote Windows.
2. Activer WSL2 :
   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --set-default-version 2
   ```

### 5.2 Côté WSL2 (Ubuntu)

Suivre les étapes **4.2, 4.3, 4.4** ci-dessus. **Ne pas** installer de pilote NVIDIA dans WSL2 (section 4.1 à sauter).

> Pour les fichiers longs (datasets Windows), voir [`WINDOWS_LONG_PATHS.md`](./WINDOWS_LONG_PATHS.md).

---

## 6. Vérification de l'installation

### 6.1 Script de diagnostic fourni

```bash
cd src/efficientnet_lite_gpu
python -m tools.hardware_test
```

Ce script appelle `tools_nvidia_cuda.py` et vérifie :
- Présence et version du driver NVIDIA (`nvidia-smi`)
- Présence de `nvcc` (CUDA Toolkit)
- Infos de build TensorFlow (CUDA / cuDNN intégrés)
- Détection du GPU par TensorFlow

### 6.2 Test TensorFlow manuel

```python
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Test d'un calcul sur GPU
with tf.device('/GPU:0'):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    c = tf.matmul(a, b)
    print("Matmul OK — shape:", c.shape)
```

Sortie attendue :
```
TF version: 2.15.x
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Built with CUDA: True
Matmul OK — shape: (1024, 1024)
```

---

## 7. Lancement du pipeline d'entraînement

```bash
cd src/efficientnet_lite_gpu
source .venv-efficientnet/bin/activate

# Vérification GPU
python -m tools.hardware_test

# Entraînement complet
python -m train.train

# Ou script autonome
python -m train.run_train
```

---

## 8. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `GPUs: []` | cuDNN non trouvé | Vérifier `LD_LIBRARY_PATH`, recopier `libcudnn*` dans `/usr/local/cuda-12.3/lib64/` |
| `Could not load dynamic library 'libcudart.so.12'` | CUDA non dans le `PATH` | Re-`source ~/.bashrc`, vérifier `ldconfig -p \| grep cuda` |
| `CUDA error: out of memory` | VRAM saturée | Réduire `batch_size` dans `config.yaml`, ou activer `tf.config.experimental.set_memory_growth` |
| `nvidia-smi` OK mais TF ignore le GPU | Mismatch TF / CUDA | Re-vérifier la matrice en section 3 |
| Sur WSL2 : `nvidia-smi: command not found` | Driver Windows obsolète | Mettre à jour le driver NVIDIA côté Windows (≥ 525) |

---

## 9. Environnement de référence validé

Configuration réelle du laptop de développement (avril 2026) :

| Composant | Version |
|---|---|
| OS | Ubuntu 22.04 LTS (WSL2 sur Windows 11) |
| GPU | **NVIDIA GeForce RTX 3070 Laptop (8 GB VRAM)** |
| Architecture CUDA | Ampere (Compute Capability 8.6) |
| Pilote NVIDIA | 545+ (supportant CUDA 12.3) |
| **CUDA Toolkit** | **12.3** |
| cuDNN | 8.9.x pour CUDA 12.x |
| Python | 3.11 |
| TensorFlow | 2.15+ |

> ℹ️ **Note sur CUDA 12.3** : la matrice officielle de TensorFlow liste CUDA 12.2 pour TF 2.15, mais CUDA 12.3 est **rétro-compatible binaire** avec 12.2 grâce au CUDA Forward Compatibility package. En pratique, TF 2.15 fonctionne sur CUDA 12.3 sans modification. Pour TF ≥ 2.16, CUDA 12.3 est directement supporté.

### 9.1 Considérations VRAM sur RTX 3070 Laptop (8 GB)

Avec 8 GB de VRAM, les contraintes pratiques sur ce pipeline :

| Configuration | Batch size viable |
|---|---|
| MobileNetV2-0.35, 224×224 | 64 (confortable) |
| MobileNetV2-1.00, 224×224 | 32 |
| EfficientNet-B0, 224×224 | 32 (la config par défaut, `batch_size: 32`) |
| EfficientNet-B3, 300×300 | 8 – 16 (OOM probable au-delà) |

Activer `tf.config.experimental.set_memory_growth` pour éviter que TF ne réserve toute la VRAM au démarrage :

```python
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 10. Références

- Matrice de compatibilité TF ↔ CUDA : <https://www.tensorflow.org/install/source#gpu>
- CUDA Toolkit Archive : <https://developer.nvidia.com/cuda-toolkit-archive>
- cuDNN : <https://developer.nvidia.com/cudnn>
- WSL2 + CUDA : <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- Équivalent AMD (serveur) : [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md)
