import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .masks import read_image, read_mask


FIELDS = ["image_id", "image_path", "mask_path", "width", "height", "group_id"]


def index_files(directory, suffix):
    if not directory.is_dir():
        raise ValueError(f"Missing directory: {directory}")
    index = {}
    for path in sorted(directory.rglob(f"ISIC_*{suffix}")):
        image_id = path.name.removesuffix(suffix)
        if not re.fullmatch(r"ISIC_\d{7}", image_id):
            continue
        if image_id in index:
            raise ValueError(f"Duplicate {image_id} in {directory}")
        index[image_id] = path.resolve()
    return index


def load_manifest(directory):
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as file:
        return {row["image_id"]: row for row in csv.DictReader(file)}


def read_split(path):
    ids = path.read_text(encoding="utf-8-sig").split()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def validate_splits(directory, manifest):
    splits = {name: read_split(directory / f"{name}.txt") for name in ("train", "val", "test")}
    if not all(splits.values()):
        raise ValueError("The experiment requires nonempty train, validation, and test splits.")
    ids = [image_id for split in splits.values() for image_id in split]
    if len(ids) != len(manifest) or set(ids) != set(manifest):
        raise ValueError("Split IDs must cover the manifest without overlap.")
    groups = [{manifest[image_id]["group_id"] for image_id in split} for split in splits.values()]
    if sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError("The same group occurs in more than one partition.")
    return splits


def group_images(fingerprints, metadata):
    labels = {image_id: image_id for image_id in fingerprints}
    seen_keys = {}

    def root(image_id):
        while labels[image_id] != image_id:
            image_id = labels[image_id]
        return image_id

    for image_id, fingerprint in fingerprints.items():
        keys = [("pixels", fingerprint)]
        if image_id in metadata:
            keys.append(("metadata", metadata[image_id]))
        for key in keys:
            peer = seen_keys.setdefault(key, image_id)
            labels[root(image_id)] = root(peer)
    return {image_id: root(image_id) for image_id in fingerprints}


def make_splits(groups, author_splits=None, training_ids=()):
    by_group = defaultdict(list)
    for image_id, group_id in sorted(groups.items()):
        by_group[group_id].append(image_id)
    shuffled = list(by_group.values())
    random.Random(42).shuffle(shuffled)
    reserved = set(training_ids)
    if not reserved <= groups.keys():
        raise ValueError("A reserved training ID is missing from the dataset.")
    reserved_groups = {groups[image_id] for image_id in reserved}

    def reserve(pool, target):
        selected, remaining = [], []
        for group in pool:
            if groups[group[0]] not in reserved_groups and len(selected) + len(group) <= target:
                selected.extend(group)
            else:
                remaining.append(group)
        if not selected:
            raise ValueError("Grouping leaves too few independent images for the requested split.")
        return selected, remaining

    if author_splits:
        paper_train = set(read_split(author_splits / "train.txt"))
        paper_test = set(read_split(author_splits / "test.txt"))
        if (len(paper_train), len(paper_test)) != (2074, 520) or paper_train | paper_test != set(groups):
            raise ValueError("Author lists must cover all 2,594 IDs without overlap.")
        if any(paper_train.intersection(group) and paper_test.intersection(group) for group in shuffled):
            raise ValueError("Author split crosses a group. Prepare a new seed-42 split instead.")
        if paper_test & reserved:
            raise ValueError("An author test image was used for debugging. Use a new seed-42 split.")
        test = list(paper_test)
        pool = [group for group in shuffled if group[0] in paper_train]
    else:
        test, pool = reserve(shuffled, round(len(groups) * 520 / 2594))
    val, pool = reserve(pool, round(len(groups) * 207 / 2594))
    train = [image_id for group in pool for image_id in group]
    if not train:
        raise ValueError("Grouping leaves no training images.")
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def manifest_path(path, directory):
    try:
        return Path(os.path.relpath(path, directory)).as_posix()
    except ValueError:  # Windows paths on different drives.
        return path.as_posix()


def prepare(images_dir, masks_dir, output, groups_csv=None, author_splits=None, smoke=False, training_ids=()):
    images = index_files(images_dir, ".jpg")
    masks = index_files(masks_dir, "_segmentation.png")
    if images.keys() != masks.keys():
        raise ValueError(f"Missing masks: {sorted(images.keys() - masks.keys())[:5]}. "
                         f"Masks without images: {sorted(masks.keys() - images.keys())[:5]}.")
    expected = 4 if smoke else 2594
    if len(images) != expected:
        raise ValueError(f"Expected {expected} labeled pairs, found {len(images)}.")
    if len({path.parent for path in images.values()}) != 1:
        raise ValueError("Put all input JPEGs in one directory.")

    metadata = {}
    if groups_csv:
        with groups_csv.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not {"image_id", "group_id"} <= set(reader.fieldnames or []):
                raise ValueError("Group CSV must have image_id and group_id columns.")
            for row in reader:
                group_id = (row["group_id"] or "").strip()
                if row["image_id"] in images and group_id:
                    previous = metadata.setdefault(row["image_id"], group_id)
                    if previous != group_id:
                        raise ValueError(f"Conflicting group IDs for {row['image_id']}")

    output = output.resolve()
    records, fingerprints = [], {}
    normalized = output / "masks"
    normalized.mkdir(parents=True, exist_ok=True)
    for image_id, image_path in images.items():
        mask_path = normalized / f"{image_id}.png"
        try:
            image = read_image(image_path)
            mask = read_mask(masks[image_id])
        except (OSError, ValueError) as error:
            raise ValueError(f"{image_id}: {error}") from error
        height, width = image.shape[:2]
        if mask.shape != (height, width):
            raise ValueError(f"{image_id}: image and mask dimensions differ.")
        fingerprints[image_id] = (height, width, hashlib.sha256(image).digest())
        if mask_path.exists():
            with Image.open(mask_path) as saved_mask:
                if not np.array_equal(np.asarray(saved_mask), mask):
                    raise ValueError(f"{image_id}: prepared mask changed. Choose a new output directory.")
        else:
            temp_path = mask_path.with_suffix(".tmp")
            Image.fromarray(mask.astype("uint8")).save(temp_path, format="PNG")
            temp_path.replace(mask_path)
        records.append(dict(image_id=image_id, image_path=manifest_path(image_path, output),
                            mask_path=manifest_path(mask_path, output), width=width, height=height))

    groups = group_images(fingerprints, metadata)
    splits = {"train": sorted(images)} if smoke else make_splits(groups, author_splits, training_ids)
    for split in ("train", "val", "test"):
        path = output / f"{split}.txt"
        if path.exists() and read_split(path) != splits.get(split):
            raise ValueError(f"Frozen {path.name} changed. Choose a new output directory.")
    for split, ids in splits.items():
        (output / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(dict(row, group_id=groups[row["image_id"]]) for row in records)
    summary = {
        "protocol": "smoke" if smoke else ("author train/test with validation" if author_splits else "seed 42"),
        "counts": {split: len(ids) for split, ids in splits.items()},
        "images_with_group_metadata": len(metadata),
        "duplicate_images": len(fingerprints) - len(set(fingerprints.values())),
    }
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Pair, audit, and split ISIC Task 1 images and masks.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--author-splits", type=Path)
    parser.add_argument("--training-ids", type=Path, help="Keep previously inspected images in training.")
    parser.add_argument("--smoke", action="store_true", help="Prepare exactly four pairs for debugging.")
    args = parser.parse_args()
    if args.smoke and (args.author_splits or args.training_ids):
        parser.error("Author splits and reserved training IDs apply only to the full dataset.")
    try:
        training_ids = read_split(args.training_ids) if args.training_ids else ()
        print(prepare(args.images, args.masks, args.output, args.groups_csv, args.author_splits,
                      args.smoke, training_ids))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
