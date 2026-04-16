#!/usr/bin/env python3
"""
Push model artifacts to HuggingFace Hub.

Prerequisites:
    huggingface-cli login   # or set HF_TOKEN env var
    pip install huggingface-hub

Usage:
    cd src/mobilenet_v2_small
    python -m tools.push_to_hub
    python -m tools.push_to_hub --repo-id your-org/your-model
    python -m tools.push_to_hub --private   # private repo
"""

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "whispr/mobilenetv2-food-classifier"
EXPORTS_DIR = "exports"


def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--exports-dir", type=str, default=EXPORTS_DIR,
                        help=f"Exports directory (default: {EXPORTS_DIR})")
    parser.add_argument("--private", action="store_true",
                        help="Create private repo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    if not exports_dir.exists():
        print(f"ERROR: exports dir not found: {exports_dir}")
        print("Run `python -m tools.convert_model` first.")
        sys.exit(1)

    # Verify required files exist.
    required = ["README.md", "labels.json", "config.json"]
    missing = [f for f in required if not (exports_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing files in {exports_dir}: {missing}")
        sys.exit(1)

    # Only upload model-essential files.
    INCLUDE = {
        "BestModelMobileNetV2.keras",
        "README.md",
        "labels.json",
        "config.json",
    }
    INCLUDE_DIRS = {"tflite", "tfjs"}

    files = []
    for p in sorted(exports_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("_"):
            continue
        rel = p.relative_to(exports_dir)
        # Include if file is in whitelist or under an allowed subdir.
        if rel.name in INCLUDE or (rel.parts[0] in INCLUDE_DIRS if len(rel.parts) > 1 else False):
            files.append((p, rel, p.stat().st_size))

    print(f"Repo:    {args.repo_id}")
    print(f"Source:  {exports_dir}/")
    print(f"Files:   {len(files)}")
    print()

    total_size = 0
    for p, rel, size in files:
        if size > 1024 * 1024:
            print(f"  {rel}  ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {rel}  ({size / 1024:.1f} KB)")
        total_size += size

    print(f"\nTotal: {total_size / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the above files. Exiting.")
        return

    # Import and verify login.
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    api = HfApi()

    try:
        user_info = api.whoami()
        print(f"\nLogged in as: {user_info.get('name', user_info.get('fullname', 'unknown'))}")
    except Exception:
        print("\nERROR: Not logged in to HuggingFace.")
        print("Run one of:")
        print("  huggingface-cli login")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    # Create repo if needed.
    print(f"\nCreating/verifying repo: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload entire directory.
    print(f"Uploading {len(files)} files...")
    api.upload_folder(
        folder_path=str(exports_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload MobileNetV2 food classifier (Keras + TFLite + TFJS)",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
