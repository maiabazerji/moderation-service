import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_MERGE_MAP = {
    "Donuts": "Donut",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Clean cross-class duplicate images by content hash.")
    parser.add_argument("--input-dir", required=True, type=str, help="Class-based raw dataset directory.")
    parser.add_argument("--output-dir", required=True, type=str, help="Cleaned raw dataset output directory.")
    parser.add_argument("--quarantine-dir", required=True, type=str, help="Where removed cross-class duplicates are stored.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    quarantine_dir = Path(args.quarantine_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    class_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError("No class folders found in input dir.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if quarantine_dir.exists():
        shutil.rmtree(quarantine_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Rule: keep the first merged class in alphabetical order for a given hash.
    hash_records = defaultdict(list)
    for class_dir in class_dirs:
        merged_class = CLASS_MERGE_MAP.get(class_dir.name, class_dir.name)
        for p in sorted(class_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            h = _sha256_file(p)
            hash_records[h].append((merged_class, p))

    moved = []
    kept = 0
    for h, items in hash_records.items():
        owner_class = sorted([x[0] for x in items])[0]
        owner_candidates = [x for x in items if x[0] == owner_class]
        owner_src = sorted(owner_candidates, key=lambda x: str(x[1]))[0][1]

        dst_owner_dir = output_dir / owner_class
        dst_owner_dir.mkdir(parents=True, exist_ok=True)
        dst_owner = dst_owner_dir / owner_src.name
        if dst_owner.exists():
            stem, suf = dst_owner.stem, dst_owner.suffix
            idx = 1
            while True:
                cand = dst_owner_dir / f"{stem}__dup{idx}{suf}"
                if not cand.exists():
                    dst_owner = cand
                    break
                idx += 1
        shutil.copy2(owner_src, dst_owner)
        kept += 1

        for cls, src in items:
            if src == owner_src:
                continue
            q_dir = quarantine_dir / cls
            q_dir.mkdir(parents=True, exist_ok=True)
            q_dst = q_dir / src.name
            if q_dst.exists():
                stem, suf = q_dst.stem, q_dst.suffix
                idx = 1
                while True:
                    cand = q_dir / f"{stem}__dup{idx}{suf}"
                    if not cand.exists():
                        q_dst = cand
                        break
                    idx += 1
            shutil.copy2(src, q_dst)
            moved.append(
                {
                    "hash": h,
                    "kept_class": owner_class,
                    "kept_path": str(owner_src),
                    "removed_class": cls,
                    "removed_path": str(src),
                    "quarantine_path": str(q_dst),
                }
            )

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "quarantine_dir": str(quarantine_dir),
        "unique_hash_kept": kept,
        "cross_class_duplicates_quarantined": len(moved),
        "examples": moved[:50],
    }
    with open(output_dir / "cross_class_cleanup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
