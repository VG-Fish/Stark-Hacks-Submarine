"""Extract a curated 3,500-image subset from Dataset/ into SubDataset/ with per-prefix quotas
and train/val/test splits.

Target distribution (validation carved from train):
  Prefix        Target  Test(15%)
  DeepCrack       530       80
  CFD             118       18
  cracktree200    200       30
  noncrack        350       52
  CRACK500      2,302      345
  TOTAL         3,500      525

If a prefix has fewer images than its target, all available images are used
and the shortfall is filled with random images from other available prefixes.
"""

import random
import shutil
from pathlib import Path

SEED = 42
TOTAL_TARGET = 3500
SRC_DIR = Path("Dataset")
DST_DIR = Path("SubDataset")

# (prefix, desired_total, test_count)
QUOTAS: list[tuple[str, int, int]] = [
    ("DeepCrack",    530,  80),
    ("CFD",          118,  18),
    ("cracktree200", 200,  30),
    ("noncrack",     350,  52),
    ("CRACK500",    2302, 345),
]

TEST_RATIO = 0.15
VAL_RATIO = 0.10


def collect_pairs_by_prefix(src: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Collect all (image, mask) pairs grouped by filename prefix."""
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for split in ("train", "test"):
        img_dir = src / split / "images"
        mask_dir = src / split / "masks"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue
            prefix = img_path.stem.split("_")[0].split("-")[0]
            groups.setdefault(prefix, []).append((img_path, mask_path))
    return groups


def copy_pairs(pairs: list[tuple[Path, Path]], dst_images: Path, dst_masks: Path) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    for img_path, mask_path in pairs:
        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(mask_path, dst_masks / mask_path.name)


def main() -> None:
    rng = random.Random(SEED)
    all_groups = collect_pairs_by_prefix(SRC_DIR)

    print("Available pairs per prefix:")
    for prefix in sorted(all_groups):
        print(f"  {prefix}: {len(all_groups[prefix])}")
    print()

    quota_prefixes = {p for p, _, _ in QUOTAS}
    used_ids: set[str] = set()  # track image names already sampled
    sampled_all: list[tuple[Path, Path]] = []
    shortfall = 0

    # --- Phase 1: sample from each quota prefix (capped at available) ---
    for prefix, desired, n_test in QUOTAS:
        available = all_groups.get(prefix, [])
        actual = min(desired, len(available))
        if actual < desired:
            print(f"  {prefix}: wanted {desired}, only {len(available)} available — using all")
            shortfall += desired - actual

        sampled = rng.sample(available, actual)
        for img, _ in sampled:
            used_ids.add(img.name)
        sampled_all.extend(sampled)

    # --- Phase 2: fill shortfall from other prefixes ---
    if shortfall > 0:
        extras: list[tuple[Path, Path]] = []
        for prefix in sorted(all_groups):
            if prefix in quota_prefixes:
                continue
            for pair in all_groups[prefix]:
                if pair[0].name not in used_ids:
                    extras.append(pair)
        rng.shuffle(extras)
        fill = extras[:shortfall]
        print(f"\nFilling {len(fill)} shortfall images from other prefixes")
        sampled_all.extend(fill)

    # --- Phase 3: split into train / val / test ---
    rng.shuffle(sampled_all)

    # Maintain per-prefix test ratios from QUOTAS; for extras use TEST_RATIO
    quota_test = {p: t for p, _, t in QUOTAS}
    train_all: list[tuple[Path, Path]] = []
    val_all: list[tuple[Path, Path]] = []
    test_all: list[tuple[Path, Path]] = []

    # Re-group sampled images by prefix
    sampled_by_prefix: dict[str, list[tuple[Path, Path]]] = {}
    for pair in sampled_all:
        prefix = pair[0].stem.split("_")[0].split("-")[0]
        sampled_by_prefix.setdefault(prefix, []).append(pair)

    for prefix, pairs in sampled_by_prefix.items():
        rng.shuffle(pairs)
        n = len(pairs)
        if prefix in quota_test:
            n_test = min(quota_test[prefix], n)
        else:
            n_test = round(n * TEST_RATIO)

        test_pairs = pairs[:n_test]
        remainder = pairs[n_test:]
        n_val = round(len(remainder) * VAL_RATIO / (1 - VAL_RATIO))
        n_val = min(n_val, len(remainder))
        val_pairs = remainder[:n_val]
        train_pairs = remainder[n_val:]

        print(f"{prefix:15s}  total={n:>5d}  train={len(train_pairs):>5d}  "
              f"val={len(val_pairs):>4d}  test={len(test_pairs):>4d}")

        train_all.extend(train_pairs)
        val_all.extend(val_pairs)
        test_all.extend(test_pairs)

    grand_total = len(train_all) + len(val_all) + len(test_all)
    print(f"\n{'TOTAL':15s}  total={grand_total:>5d}  train={len(train_all):>5d}  "
          f"val={len(val_all):>4d}  test={len(test_all):>4d}")

    # Shuffle within each split so prefixes are interleaved
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    # Write to disk
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    for split_name, pairs in [("train", train_all), ("val", val_all), ("test", test_all)]:
        copy_pairs(pairs, DST_DIR / split_name / "images", DST_DIR / split_name / "masks")
        print(f"  Copied {split_name}/")

    print(f"\nDone → {DST_DIR}/")


if __name__ == "__main__":
    main()
