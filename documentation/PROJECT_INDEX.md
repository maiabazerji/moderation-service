# Project Documentation Index

Centralised list of the docs under `documentation/`. Grouped by what you're trying to do, not by file type.

## Active module

- Main module: `src/mobilenet_v2_small`
- CLI entry point: `src/mobilenet_v2_small/main.py` (`--action train | eval | validation | test`)
- Direct train entry point: `python -m train.train` (reads `config.yaml`)

## Starting a run

- Full workflow (data → train → export → deploy): `PIPELINE.md`
- Dataset prep (fetch, clean, split): `DATASET_PREPARATION.md`
- Hyperparameter sweep: `HYPERPARAMETER_SWEEP_GUIDE.md`

## Environment

- Laptop (NVIDIA RTX 3070, CUDA 12.3): `SETUP_CUDA_NVIDIA.md`
- Server (AMD 2× RX 6600 XT, ROCm 7.2.1, Debian 12): `SETUP_ROCM_AMD.md`
- Windows long paths: `WINDOWS_LONG_PATHS.md`

## Results and exports

- Last production run (MobileNetV2-0.35, French): `RAPPORT_ENTRAINEMENT_MOBILENETV2.md`
- TensorFlow.js conversion: `TFJS_CONVERSION_README.md`

## Project conventions and history

- `1_architecture/1_system_design.md` — high-level design
- `STRUCTURE_CONVENTIONS.md` — folder / naming rules
- `DEPRECATION_MAP.md` — legacy / duplicated scripts tracked here

## Dependencies

- Runtime: `src/mobilenet_v2_small/requirements.txt`
- Fetch-only (no TF): `src/mobilenet_v2_small/requirements-fetch-only.txt`
