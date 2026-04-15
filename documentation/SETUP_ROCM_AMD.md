# Environnement GPU AMD (ROCm) — serveur

Notes pratiques pour faire tourner `src/efficientnet_lite_gpu/` sur le **serveur AMD**. Le laptop de dev utilise une carte NVIDIA et est documenté séparément dans [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md) ; il faut valider les changements sur les deux cibles avant une release.

Le code applicatif (`train/`, `tools/`) est identique : ROCm se branche au niveau du package TensorFlow et du runtime `/opt/rocm`, pas du code Python.

## Le serveur en place

Ce qu'il y a vraiment sur la machine d'entraînement (mesuré avec `lspci`, `rocminfo`, `rocm-smi`, `free`, `df`) :

- **GPU** : 2× AMD Radeon RX 6600 XT, Navi 23, rapporté par ROCm comme `gfx1030` via l'override (cf Variables), 8 GB VRAM chacune.
- **OS** : Debian 12 bookworm, kernel 6.1.0-21-amd64. Debian n'est pas officiellement supporté par ROCm ; on s'appuie sur les paquets Ubuntu jammy qui passent en pratique.
- **CPU / RAM** : Intel Xeon E5-2699 v3 (18C/36T @ 2.30 GHz), 15 GB de RAM. Les 15 GB sont le point faible — `dataset_raw_*` font ~2 GB chacun et TF charge beaucoup de cache, surveiller `free -h` pendant les sweeps.
- **Stockage** : 456 GB sur `/`, à 87 % d'occupation (57 GB libres au moment où j'écris). Les `dataset_raw_*/` et `results_sweep/` sont les plus gros consommateurs, d'où leur présence dans `.gitignore`.
- **ROCm / TF** : ROCm 7.2.1 (`cat /opt/rocm/.info/version`), TensorFlow **stock** 2.19.0 (pas `tensorflow-rocm`), Python 3.11.2.

La RX 6600 XT n'est pas dans la matrice de support officielle AMD (ROCm liste les RX 6800+ / 7900+ et les Instinct). Elle fonctionne parce qu'on force `HSA_OVERRIDE_GFX_VERSION=10.3.0`, qui fait passer le binaire pour du gfx1030 (RDNA2 générique). Si on repasse en prod "officielle", il faudra migrer vers une carte supportée (MI210, W7800). En attendant, les runs passent mais on est hors matrice.

ROCm ne supporte pas Windows pour l'entraînement TF — le serveur doit être Linux natif, pas de WSL2.

## Compatibilité TF / ROCm — à quoi se fier

Il y a un vrai virage entre ROCm 5.x et ROCm 6+/7+ :

- **ROCm ≤ 5.7** : AMD distribuait une wheel séparée `tensorflow-rocm` (ex. `tensorflow-rocm==2.14.0`). Il fallait l'installer **avant** les autres dépendances sinon `pip install tensorflow` écrasait tout.
- **ROCm 6+/7+** : le paquet stock `tensorflow` détecte `/opt/rocm` et route via HIP. Plus besoin de wheel séparée. C'est la voie qu'on utilise sur le serveur (`pip show tensorflow` retourne `2.19.0`, GPU détecté, terminé).

Repères de compatibilité que j'ai retenus — pas une liste exhaustive, juste ce qui est sûr dans le contexte actuel :

| TensorFlow | ROCm | Commentaire |
|---|---|---|
| 2.19.x | 7.x / 6.x | ce qu'on utilise |
| 2.16 – 2.18 | 6.x | OK, stock tensorflow |
| 2.14 – 2.15 | 5.7 | via `tensorflow-rocm` |
| ≤ 2.13 | ≤ 5.6 | legacy, à éviter |

Pour une config différente, la source unique fiable reste la page d'install ROCm (`https://rocm.docs.amd.com/projects/install-on-linux/en/latest/`). Les tags TF/ROCm dans la doc AMD peuvent être en retard sur PyPI, vérifier les deux.

## Installer ROCm sur le serveur

Le package `amdgpu-install` cible officiellement Ubuntu 22.04 jammy. Sur notre Debian 12 il passe en forçant le .deb Ubuntu :

```bash
sudo apt update
sudo apt install -y wget gnupg2 curl lsb-release

wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/jammy/amdgpu-install_7.2.60201-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.60201-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

Le `--no-dkms` est fait exprès : Debian 12 a déjà un pilote `amdgpu` dans le kernel 6.1, on n'a pas besoin que ROCm recompile un module DKMS.

Dépendances qui peuvent râler côté Debian : `lsb-release`, `libtinfo5`. `sudo apt --fix-broken install` règle à peu près tous les cas que j'ai vus.

Après install :

```bash
cat /opt/rocm/.info/version          # 7.2.1
rocminfo | grep -E "Name:.*gfx"      # 2× Name: gfx1030
rocm-smi --showproductname            # 2× RX 6600 XT
```

L'utilisateur qui lance les runs doit être dans les groupes `render` et `video` (droits sur `/dev/kfd` et `/dev/dri`), sinon `HSA Error: Permission denied`. `sudo usermod -aG render,video $USER` puis se reconnecter.

## Environnement Python

```bash
cd src/efficientnet_lite_gpu
python3.11 -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # tensorflow stock suffit (voir section précédente)
```

Pour un serveur ROCm ≤ 5.7 l'ordre change :

```bash
pip install tensorflow-rocm==2.14.0
grep -v '^tensorflow' requirements.txt | pip install -r /dev/stdin
```

## Variables d'environnement (les plus utiles)

À placer dans `~/.bashrc` ou sourcées par un script avant `python -m train.train`. Celles qui comptent vraiment sur notre serveur :

```bash
# Indispensable sur RX 6600 XT (Navi 23 → gfx1030 générique)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Restreindre la visibilité GPU — équivalent de CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0           # un seul GPU
# export HIP_VISIBLE_DEVICES=0,1       # les deux, nécessite MirroredStrategy explicite dans le code

# Debug verbeux (désactivé par défaut)
# export AMD_LOG_LEVEL=3
```

La variable `LD_LIBRARY_PATH` doit inclure `/opt/rocm/lib` — l'installeur `amdgpu-install` le fait déjà via `/etc/ld.so.conf.d/`. Vérifier avec `ldconfig -p | grep rocm`.

## Valider que tout marche

```python
import tensorflow as tf
print(tf.__version__)                                   # 2.19.0
print(tf.config.list_physical_devices('GPU'))           # doit lister 1 ou 2 GPUs
with tf.device('/GPU:0'):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    print(tf.matmul(a, b).shape)                        # (1024, 1024)
```

Si ça renvoie `GPUs: []` : cas le plus fréquent, c'est `HSA_OVERRIDE_GFX_VERSION` qui n'est pas exporté dans le shell en cours. Un `echo $HSA_OVERRIDE_GFX_VERSION` doit afficher `10.3.0`.

`tools/hardware_test.py` appelle `nvidia-smi` (il est NVIDIA-first) et affichera des warnings côté AMD — c'est inoffensif, ce qui compte est juste que `tf.config.list_physical_devices('GPU')` soit non vide. Un refactor pour ajouter `rocm-smi` serait bienvenu mais pas urgent.

## Lancer un entraînement

Aucune particularité côté code :

```bash
cd src/efficientnet_lite_gpu
source .venv-rocm/bin/activate
python -m train.train
```

Si la VRAM sature avec `batch_size: 32` : soit réduire dans `config.yaml`, soit ajouter dans le code au démarrage

```python
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

pour laisser TF allouer incrémentalement au lieu de réserver toute la VRAM d'un coup.

## Images Docker (option propre pour prod)

AMD publie des images officielles sur `hub.docker.com/r/rocm/tensorflow`. Les tags changent régulièrement, ne pas en coder en dur dans nos scripts — aller chercher le tag à jour pour la version ROCm cible.

```bash
# Récupérer un tag à jour pour ROCm 7.x
docker pull rocm/tensorflow:latest     # ou un tag versionné depuis le registry

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size 16G \
  -v $(pwd):/workspace \
  rocm/tensorflow:latest bash
```

Dans le conteneur : `pip install -r requirements.txt` puis `python -m train.train`. L'image embarque déjà un TF compatible ROCm.

## Dépannage — les cas qu'on a vus

| Symptôme | Cause typique | Ce qui débloque |
|---|---|---|
| `rocm-smi` vide / erreur | Pilote `amdgpu` pas chargé | `lsmod \| grep amdgpu` ; `sudo modprobe amdgpu` ; reboot |
| `GPUs: []` côté TF | Shell sans `HSA_OVERRIDE_GFX_VERSION` | sourcer `.bashrc` ou `export` à la main |
| `HSA Error: No GPU devices found` | GPU pas reconnu | vérifier la gfx target, ajuster l'override |
| `Permission denied` sur `/dev/kfd` | Pas dans `render`/`video` | `sudo usermod -aG render,video $USER` + relogin |
| OOM à batch 32 | VRAM insuffisante | réduire `batch_size`, activer `set_memory_growth` |
| Entraînement qui tombe sur CPU | ROCm pas initialisé | `AMD_LOG_LEVEL=3 python ...` pour voir pourquoi |
| `tensorflow-rocm` écrasé par un `pip install tensorflow` | Ordre d'install cassé | `pip uninstall -y tensorflow tensorflow-rocm` puis réinstaller dans le bon ordre (legacy) |

## CUDA / ROCm — les différences qu'on manipule

| Aspect | CUDA (laptop) | ROCm (serveur) |
|---|---|---|
| Paquet TF | `tensorflow` | `tensorflow` (stock) depuis ROCm 6+, `tensorflow-rocm` sur legacy ≤ 5.7 |
| Monitoring | `nvidia-smi` | `rocm-smi`, `rocminfo` |
| Visibilité GPU | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` |
| Override arch | — | `HSA_OVERRIDE_GFX_VERSION` |
| Windows | OK via WSL2 | non supporté |
| Image Docker | `tensorflow/tensorflow:*-gpu` | `rocm/tensorflow:*` |

Le code `src/efficientnet_lite_gpu/` ne doit pas brancher sur l'un ou l'autre ; si un bug apparaît seulement sur une cible, c'est un problème de runtime/lib, pas d'API.

## Liens qu'on consulte régulièrement

- Install ROCm Linux : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>
- Matrice de support GPU : <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>
- Registry Docker officiel AMD : <https://hub.docker.com/r/rocm/tensorflow>
- TensorFlow upstream ROCm : <https://github.com/ROCm/tensorflow-upstream>
- Équivalent côté laptop NVIDIA : [`SETUP_CUDA_NVIDIA.md`](./SETUP_CUDA_NVIDIA.md)
