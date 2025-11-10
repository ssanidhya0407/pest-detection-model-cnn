#!/usr/bin/env python3
"""
Ingest new images into the dataset folder structure used by training.

Usage:
  python scripts/ingest_new_data.py --source new_images --dest pest --train-ratio 0.8

Expectations:
- `source` should contain one folder per class, e.g. new_images/armyworm/*.jpg
- The script will copy files into `dest/train/<class>/` and `dest/test/<class>/` with a reproducible split.
"""
import argparse
from pathlib import Path
import random
import shutil
import uuid
import sys


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def is_image(p: Path):
    return p.suffix.lower() in IMAGE_EXTENSIONS


def unique_name(dest_dir: Path, original_name: str):
    # Ensure we don't collide with existing files: append uuid4 if needed
    dest = dest_dir / original_name
    if not dest.exists():
        return original_name
    base = Path(original_name).stem
    ext = Path(original_name).suffix
    for _ in range(10):
        candidate = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
        if not (dest_dir / candidate).exists():
            return candidate
    # fallback
    return f"{base}_{uuid.uuid4().hex}{ext}"


def ingest(source: Path, dest: Path, train_ratio: float = 0.8, move: bool = False, seed: int = 42):
    if not source.exists():
        raise SystemExit(f"Source folder {source} does not exist")
    dest = dest.resolve()
    train_root = dest / 'train'
    test_root = dest / 'test'
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # If source contains class subfolders, use them.
    classes = [p for p in source.iterdir() if p.is_dir()]
    if classes:
        # behave as before
        total_copied = 0
        for class_dir in classes:
            class_name = class_dir.name
            files = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
            if not files:
                print(f"[INFO] No images found for class {class_name} in {class_dir}, skipping")
                continue
            random.shuffle(files)
            split_idx = int(len(files) * train_ratio)
            train_files = files[:split_idx]
            test_files = files[split_idx:]

            target_train = train_root / class_name
            target_test = test_root / class_name
            target_train.mkdir(parents=True, exist_ok=True)
            target_test.mkdir(parents=True, exist_ok=True)

            for src in train_files:
                dest_name = unique_name(target_train, src.name)
                dst = target_train / dest_name
                if move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                total_copied += 1

            for src in test_files:
                dest_name = unique_name(target_test, src.name)
                dst = target_test / dest_name
                if move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                total_copied += 1

            print(f"[INFO] Class {class_name}: train {len(train_files)} files, test {len(test_files)} files -> copied to {target_train} and {target_test}")

        print(f"[DONE] Total files copied/moved: {total_copied}")
        print(f"You can now run training: python pest_model/train.py --data-dir {dest} --epochs 30 --batch-size 16")
        return

    # No class subfolders found; caller likely provided a flat directory. We'll not handle it here.
    raise SystemExit(f"No class subfolders found in {source}. For a flat folder, re-run with the '--flat --label-from-filename' option or provide a mapping CSV.")


def ingest_flat(source: Path, dest: Path, train_ratio: float = 0.8, move: bool = False, seed: int = 42, label_from_filename: bool = False, mapping_csv: Path = None, delimiter_chars=('_','-')):
    """Ingest when source is a flat folder of images.
    If label_from_filename=True, filenames must start with '<class>_' or '<class>-' etc.
    Alternatively provide mapping_csv with two columns: filename,classname (no header).
    """
    if not source.exists():
        raise SystemExit(f"Source folder {source} does not exist")
    files = [p for p in source.iterdir() if p.is_file() and is_image(p)]
    if not files:
        raise SystemExit(f"No images found in {source}")

    # build mapping filename -> class
    mapping = {}
    if mapping_csv:
        if not mapping_csv.exists():
            raise SystemExit(f"Mapping CSV {mapping_csv} not found")
        for line in mapping_csv.read_text().splitlines():
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2: continue
            fname, cls = parts[0], parts[1]
            mapping[fname] = cls

    entries = []
    for f in files:
        cls = None
        if f.name in mapping:
            cls = mapping[f.name]
        elif label_from_filename:
            stem = f.stem
            for d in delimiter_chars:
                if d in stem:
                    cls = stem.split(d)[0]
                    break
        if cls is None:
            print(f"[WARN] Could not determine class for {f.name}; skipping. Provide mapping CSV or use --label-from-filename")
            continue
        entries.append((f, cls))

    if not entries:
        raise SystemExit("No files to ingest after attempting to determine labels.")

    random.seed(seed)
    # group by class
    by_class = {}
    for f, cls in entries:
        by_class.setdefault(cls, []).append(f)

    total = 0
    train_root = dest / 'train'
    test_root = dest / 'test'
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    for cls, flist in by_class.items():
        random.shuffle(flist)
        split_idx = int(len(flist) * train_ratio)
        train_files = flist[:split_idx]
        test_files = flist[split_idx:]
        tdir = train_root / cls
        vdir = test_root / cls
        tdir.mkdir(parents=True, exist_ok=True)
        vdir.mkdir(parents=True, exist_ok=True)
        for src in train_files:
            dest_name = unique_name(tdir, src.name)
            dst = tdir / dest_name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            total += 1
        for src in test_files:
            dest_name = unique_name(vdir, src.name)
            dst = vdir / dest_name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            total += 1
        print(f"[INFO] Class {cls}: train {len(train_files)} test {len(test_files)}")

    print(f"[DONE] Total files copied/moved: {total}")
    print(f"You can now run training: python pest_model/train.py --data-dir {dest} --epochs 30 --batch-size 16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source folder containing class subfolders (or flat images)')
    parser.add_argument('--dest', type=str, default='pest', help='Destination dataset root (default: pest)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of images to use for training (default 0.8)')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flat', action='store_true', help='Source is a flat folder of images (no class subfolders)')
    parser.add_argument('--label-from-filename', action='store_true', help="When --flat, infer class from filename prefix like 'armyworm_001.jpg' or 'armyworm-001.jpg'.")
    parser.add_argument('--mapping-csv', type=str, default=None, help='Optional CSV mapping filename,classname (for flat mode)')
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    if args.flat:
        mapping = Path(args.mapping_csv) if args.mapping_csv else None
        ingest_flat(src, dst, train_ratio=args.train_ratio, move=args.move, seed=args.seed, label_from_filename=args.label_from_filename, mapping_csv=mapping)
    else:
        ingest(src, dst, train_ratio=args.train_ratio, move=args.move, seed=args.seed)


if __name__ == '__main__':
    main()
