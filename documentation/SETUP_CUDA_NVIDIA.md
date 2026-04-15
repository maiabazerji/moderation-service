# Environnement GPU NVIDIA (CUDA) — laptop de dev

Notes pratiques pour faire tourner `src/efficientnet_lite_gpu/` sur le **laptop NVIDIA**. Le serveur d'entraînement utilise une carte AMD, documentation à part : [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md). Valider les changements de pipeline sur les deux cibles avant release.

## Le laptop en place

- **GPU** : NVIDIA GeForce RTX 3070 Laptop (Ampere, Compute Capability 8.6, 8 GB VRAM).
- **OS** : Ubuntu 22.04 LTS sous WSL2 (Windows 11) — l'OS Linux natif marche aussi.
- **CUDA / cuDNN** : CUDA Toolkit 12.3, cuDNN 8.9 (branches CUDA 12.x).
- **Python / TF** : Python 3.11, TensorFlow 2.15+.

Les 8 GB de VRAM sont la contrainte principale. Les backbones actuels passent tous sans OOM ; c'est EfficientNet-B3 à 300×300 qui commence à poser problème.

## Compatibilité TF / CUDA — ce qui a été testé

La compatibilité TF ↔ CUDA est stricte : si la combinaison n'est pas bonne, `tf.config.list_physical_devices('GPU')` renvoie `[]` sans erreur explicite, on passe 30 minutes à chercher.

Ce qui marche en pratique sur ce projet :

| TensorFlow | Python | CUDA | cuDNN | Driver NVIDIA min |
|---|---|---|---|---|
| 2.16.x | 3.9 – 3.11 | 12.3 | 8.9 | 545+ |
| 2.15.x | 3.9 – 3.11 | 12.2 (12.3 OK en forward-compat) | 8.9 | 525+ |
| 2.14.x | 3.9 – 3.11 | 11.8 | 8.7 | 450+ |

Pour une combinaison plus ancienne ou exotique, la référence fiable reste `https://www.tensorflow.org/install/source#gpu`.

Le point sur **CUDA 12.3** : la matrice officielle TF 2.15 mentionne 12.2, mais TF 2.15 tourne sans problème sur une install 12.3 grâce au CUDA Forward Compatibility. Pour TF ≥ 2.16, 12.3 est listé directement. C'est ce qu'on a sur le laptop.

## Installer sur Ubuntu 22.04 natif

**Driver NVIDIA**. Les pilotes packagés Ubuntu sont fiables :

```bash
sudo apt purge '^nvidia-.*' -y
sudo apt autoremove -y
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi   # doit afficher le GPU + la version driver + CUDA max supportée
```

**CUDA Toolkit 12.3**. Utiliser le runfile plutôt que les paquets deb (plus simple à désinstaller si besoin) :

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run --toolkit --silent --override
```

`--toolkit` évite d'écraser le driver qu'on vient d'installer. Ajouter au `~/.bashrc` :

```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

Vérifier : `nvcc --version` doit afficher `release 12.3`.

**cuDNN 8.9** (compte NVIDIA requis pour le download) :

```bash
tar -xvf cudnn-linux-x86_64-8.9.*_cuda12-archive.tar.xz
sudo cp cudnn-*/include/cudnn*.h /usr/local/cuda-12.3/include/
sudo cp cudnn-*/lib/libcudnn*     /usr/local/cuda-12.3/lib64/
sudo chmod a+r /usr/local/cuda-12.3/include/cudnn*.h /usr/local/cuda-12.3/lib64/libcudnn*
```

Vérifier la version : `grep -A 2 CUDNN_MAJOR /usr/local/cuda-12.3/include/cudnn_version.h`.

**Env Python** :

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
pip install -r requirements.txt
```

## Installer sur Windows 10/11 via WSL2

WSL2 est la voie qu'on utilise. Pas besoin de dual-boot, le GPU passe à travers.

Côté Windows :

1. Mettre à jour le driver NVIDIA GeForce (≥ 545 pour CUDA 12.3). Le driver Linux ne s'installe **pas** dans WSL2 — WSL2 utilise le driver Windows via le passthrough `/dev/dxg`.
2. ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --set-default-version 2
   ```

Côté WSL2 (Ubuntu) : suivre les étapes CUDA Toolkit + cuDNN + venv ci-dessus, **sauter la partie driver NVIDIA**. `nvidia-smi` est disponible directement dans WSL2 grâce au passthrough.

Pour les datasets avec des chemins longs (nom de fichier scrappé depuis Google), voir [`WINDOWS_LONG_PATHS.md`](./WINDOWS_LONG_PATHS.md).

## Valider que tout marche

```bash
python -m tools.hardware_test
```

Le script passe par `tools/tools_nvidia_cuda.py` et vérifie driver, `nvcc`, build info TF et détection GPU. Test manuel minimal :

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))   # au moins 1 entry
print(tf.test.is_built_with_cuda())             # True

with tf.device('/GPU:0'):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    print(tf.matmul(a, b).shape)                # (1024, 1024)
```

## Lancer un entraînement

```bash
cd src/efficientnet_lite_gpu
source .venv-efficientnet/bin/activate
python -m tools.hardware_test      # optionnel, sanity check
python -m train.train
```

Le script autonome `train/run_train.py` existe encore mais il n'utilise pas `config.yaml` proprement, éviter pour un run sérieux.

## VRAM et batch size

Sur le RTX 3070 Laptop (8 GB) avec le pipeline actuel :

| Config | Batch size qui passe |
|---|---|
| MobileNetV2-0.35, 224×224 | 64 large |
| MobileNetV2-1.00, 224×224 | 32 |
| EfficientNet-B0, 224×224 (défaut) | 32 |
| EfficientNet-B3, 300×300 | 8–16, au-delà ça OOM |

Pour éviter que TF réserve les 8 GB au démarrage :

```python
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

À appeler avant toute allocation. On ne l'a pas mis par défaut dans `train.py` pour rester proche du comportement serveur, mais c'est utile en debug local.

## Cas de panne vus en pratique

| Symptôme | Cause typique | Ce qui débloque |
|---|---|---|
| `GPUs: []` alors que `nvidia-smi` OK | cuDNN pas trouvé | vérifier `LD_LIBRARY_PATH`, re-copier les `libcudnn*` dans `lib64` |
| `Could not load dynamic library 'libcudart.so.12'` | CUDA pas dans le PATH | `source ~/.bashrc`, `ldconfig -p \| grep cuda` |
| `CUDA error: out of memory` | VRAM saturée | réduire `batch_size` ou activer `set_memory_growth` |
| `nvidia-smi` OK mais TF ignore le GPU | mismatch TF / CUDA | revoir la matrice de compatibilité plus haut |
| Sous WSL2 : `nvidia-smi: command not found` | driver Windows trop ancien | MAJ du driver NVIDIA Windows (≥ 545) |

## Liens utiles

- Matrice officielle TF ↔ CUDA : <https://www.tensorflow.org/install/source#gpu>
- CUDA Toolkit Archive : <https://developer.nvidia.com/cuda-toolkit-archive>
- cuDNN : <https://developer.nvidia.com/cudnn>
- WSL2 + CUDA : <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- Côté serveur AMD : [`SETUP_ROCM_AMD.md`](./SETUP_ROCM_AMD.md)
