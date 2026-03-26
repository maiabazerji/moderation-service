import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple

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


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}.")


def scan_raw_dataset(raw_dir: Path) -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
    # Read a single class-folder raw source and hash each file.
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in raw dataset: {raw_dir}")

    samples_by_class: Dict[str, List[Dict[str, str]]] = {}
    hash_to_classes: Dict[str, set] = defaultdict(set)
    for class_dir in class_dirs:
        merged_class_name = CLASS_MERGE_MAP.get(class_dir.name, class_dir.name)
        samples = []
        for p in sorted(class_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            content_hash = _sha256_file(p)
            samples.append(
                {
                    "source_path": str(p.resolve()),
                    "filename": p.name,
                    "class_name": merged_class_name,
                    "content_hash": content_hash,
                }
            )
            hash_to_classes[content_hash].add(merged_class_name)
        if not samples:
            raise ValueError(f"Class '{class_dir.name}' has no valid images.")
        samples_by_class.setdefault(merged_class_name, []).extend(samples)

    class_names = sorted(samples_by_class.keys())

    cross_class_hash_conflicts = {
        h: sorted(list(classes))
        for h, classes in hash_to_classes.items()
        if len(classes) > 1
    }
    if cross_class_hash_conflicts:
        preview = list(cross_class_hash_conflicts.items())[:20]
        raise ValueError(
            "Found identical image content across different classes. "
            "Please clean labels first. "
            f"Examples: {preview}"
        )
    return class_names, samples_by_class


def _compute_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    test_n = int(round(n * test_ratio))
    test_n = max(0, min(test_n, n))
    rem = n - test_n
    if rem <= 0:
        return 0, 0, test_n
    val_adj = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.0
    val_n = int(round(rem * val_adj))
    val_n = max(0, min(val_n, rem))
    train_n = rem - val_n
    return train_n, val_n, test_n


def stratified_split(
    samples_by_class: Dict[str, List[Dict[str, str]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    # Required order: split test first, then split val from remaining.
    rng = Random(seed)
    split_map = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for class_name, samples in samples_by_class.items():
        # On splitte par groupes de hash pour garantir qu'un même contenu n'apparaît jamais dans 2 splits.
        groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for item in samples:
            groups[item["content_hash"]].append(item)
        group_items = list(groups.values())
        rng.shuffle(group_items)

        total_n = sum(len(g) for g in group_items)
        train_n, val_n, test_n = _compute_counts(total_n, train_ratio, val_ratio, test_ratio)
        target = {"train": train_n, "val": val_n, "test": test_n}
        current = {"train": 0, "val": 0, "test": 0}
        bins = {"train": [], "val": [], "test": []}

        # Allocation gloutonne: respecte la contrainte de groupe hash indivisible.
        for g in sorted(group_items, key=len, reverse=True):
            remaining = {
                s: target[s] - current[s] for s in ["test", "val", "train"]
            }
            chosen = max(remaining, key=lambda s: remaining[s])
            bins[chosen].extend(g)
            current[chosen] += len(g)

        split_map["train"][class_name] = bins["train"]
        split_map["val"][class_name] = bins["val"]
        split_map["test"][class_name] = bins["test"]
    return split_map


def _overlap_report_by_key(
    split_map: Dict[str, Dict[str, List[Dict[str, str]]]],
    key: str,
) -> Dict[str, object]:
    a = {x[key] for v in split_map["train"].values() for x in v}
    b = {x[key] for v in split_map["val"].values() for x in v}
    c = {x[key] for v in split_map["test"].values() for x in v}
    ab = sorted(a & b)
    ac = sorted(a & c)
    bc = sorted(b & c)
    return {
        "train_val_duplicates": len(ab),
        "train_test_duplicates": len(ac),
        "val_test_duplicates": len(bc),
        "train_val_examples": ab[:20],
        "train_test_examples": ac[:20],
        "val_test_examples": bc[:20],
        "train_size": len(a),
        "val_size": len(b),
        "test_size": len(c),
    }


def assert_no_overlap(split_map: Dict[str, Dict[str, List[Dict[str, str]]]]) -> Dict[str, Dict[str, object]]:
    # Leakage guard using both source paths and content hashes.
    path_report = _overlap_report_by_key(split_map, "source_path")
    hash_report = _overlap_report_by_key(split_map, "content_hash")
    if (
        path_report["train_val_duplicates"]
        or path_report["train_test_duplicates"]
        or path_report["val_test_duplicates"]
        or hash_report["train_val_duplicates"]
        or hash_report["train_test_duplicates"]
        or hash_report["val_test_duplicates"]
    ):
        raise RuntimeError(
            "Duplicate leakage detected.\n"
            f"Path overlap: {path_report}\n"
            f"Hash overlap: {hash_report}"
        )
    return {"path_overlap": path_report, "hash_overlap": hash_report}


def _prepare_output(output_dir: Path, class_names: List[str], train_dir_name: str, val_dir_name: str, test_dir_name: str) -> None:
    # Clear output to avoid mixing with old split artifacts.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split_dir_name in [train_dir_name, val_dir_name, test_dir_name]:
        for class_name in class_names:
            (output_dir / split_dir_name / class_name).mkdir(parents=True, exist_ok=True)


def _materialize_and_manifest(
    split_map: Dict[str, Dict[str, List[Dict[str, str]]]],
    output_dir: Path,
    train_dir_name: str,
    val_dir_name: str,
    test_dir_name: str,
    mode: str,
) -> List[Dict[str, str]]:
    # Build manifest rows for reproducibility and audit.
    op = shutil.copy2 if mode == "copy" else shutil.move
    split_dir_map = {"train": train_dir_name, "val": val_dir_name, "test": test_dir_name}
    manifest = []
    for split in ["train", "val", "test"]:
        for class_name, rows in split_map[split].items():
            target_dir = output_dir / split_dir_map[split] / class_name
            for row in rows:
                src = Path(row["source_path"])
                dst = target_dir / src.name
                if dst.exists():
                    stem, suf = dst.stem, dst.suffix
                    idx = 1
                    while True:
                        cand = target_dir / f"{stem}__dup{idx}{suf}"
                        if not cand.exists():
                            dst = cand
                            break
                        idx += 1
                op(str(src), str(dst))
                manifest.append(
                    {
                        "source_path": row["source_path"],
                        "filename": row["filename"],
                        "class_name": row["class_name"],
                        "split": split,
                        "content_hash": row["content_hash"],
                    }
                )
    return manifest


def _save_audit_files(
    output_dir: Path,
    class_names: List[str],
    split_map: Dict[str, Dict[str, List[Dict[str, str]]]],
    overlap_report: Dict[str, Dict[str, object]],
    manifest: List[Dict[str, str]],
) -> None:
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(output_dir / "split_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "filename", "class_name", "split", "content_hash"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    stats = {
        "total_samples": len(manifest),
        "class_names": class_names,
        "per_class_total": {
            c: sum(len(split_map[s][c]) for s in ["train", "val", "test"])
            for c in class_names
        },
        "per_split_total": {
            s: sum(len(v) for v in split_map[s].values()) for s in ["train", "val", "test"]
        },
        "per_class_split_counts": {
            c: {
                "train": len(split_map["train"][c]),
                "val": len(split_map["val"][c]),
                "test": len(split_map["test"][c]),
            }
            for c in class_names
        },
        "duplicate_checks": overlap_report,
    }
    with open(output_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def build_clean_split(
    raw_input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    mode: str = "copy",
    seed: int = 42,
    train_dir_name: str = "Train",
    val_dir_name: str = "Val",
    test_dir_name: str = "Test",
) -> Dict[str, object]:
    # Atomic clean split pipeline with strict audit checks.
    validate_ratios(train_ratio, val_ratio, test_ratio)
    class_names, samples_by_class = scan_raw_dataset(raw_input_dir)
    split_map = stratified_split(samples_by_class, train_ratio, val_ratio, test_ratio, seed)
    overlap_report = assert_no_overlap(split_map)
    _prepare_output(output_dir, class_names, train_dir_name, val_dir_name, test_dir_name)
    manifest = _materialize_and_manifest(
        split_map, output_dir, train_dir_name, val_dir_name, test_dir_name, mode
    )
    _save_audit_files(output_dir, class_names, split_map, overlap_report, manifest)
    return {
        "output_dir": str(output_dir),
        "class_names": class_names,
        "overlap_report": overlap_report,
        "manifest_size": len(manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-free stratified splitter.")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-dir-name", type=str, default="Train")
    parser.add_argument("--val-dir-name", type=str, default="Val")
    parser.add_argument("--test-dir-name", type=str, default="Test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_clean_split(
        raw_input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        mode=args.mode,
        seed=args.seed,
        train_dir_name=args.train_dir_name,
        val_dir_name=args.val_dir_name,
        test_dir_name=args.test_dir_name,
    )
    print("Split completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
