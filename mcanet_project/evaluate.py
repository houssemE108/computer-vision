import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from .inference import load_model, predict
from .masks import contour_overlay, pixel_counts, read_image, read_mask, save_mask, scores
from .prepare_data import load_manifest, validate_splits


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(image, reference, prediction, path):
    tiles = [
        ("Original", Image.fromarray(image)),
        ("Reference", Image.fromarray(reference).convert("RGB")),
        ("Prediction", Image.fromarray(prediction).convert("RGB")),
        ("Contour", Image.fromarray(contour_overlay(image, prediction))),
    ]
    panel = Image.new("RGB", (4 * 322, 264), "white")
    draw = ImageDraw.Draw(panel)
    for index, (title, tile) in enumerate(tiles):
        tile.thumbnail((312, 234))
        x = index * 322 + 5
        draw.text((x, 4), title, fill="black")
        panel.paste(tile, (x, 22))
    panel.save(path)


def evaluate(config, checkpoint, data, output, device):
    import torch

    manifest = load_manifest(data)
    splits = validate_splits(data, manifest)
    test_ids, val_ids = splits["test"], splits["val"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty evaluation directory.")

    model = load_model(config, checkpoint, device)
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "predictions"
    masks_dir.mkdir()
    warmup = read_image(data / manifest[val_ids[0]]["image_path"])
    for _ in range(5):
        predict(model, warmup)

    rows = []
    total = Counter()
    for image_id in test_ids:
        pair = manifest[image_id]
        image = read_image(data / pair["image_path"])
        reference = read_mask(data / pair["mask_path"])
        if device == "cuda:0":
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = predict(model, image)
        if device == "cuda:0":
            torch.cuda.synchronize()
        latency_ms = 1000 * (time.perf_counter() - start)
        counts = pixel_counts(prediction, reference)
        total.update(counts)
        mask_path = masks_dir / f"{image_id}.png"
        save_mask(prediction, mask_path)
        rows.append(dict(image_id=image_id, prediction=mask_path.relative_to(output).as_posix(),
                         area_fraction=float(reference.mean()), latency_ms=latency_ms,
                         **counts, **scores(counts)))

    lesion = scores(total)
    background = scores(dict(tp=total["tn"], fp=total["fn"], fn=total["fp"]))
    summary = {
        "images": len(rows),
        "batch_size": 1,
        "device": torch.cuda.get_device_name() if device == "cuda:0" else "CPU",
        "resolution": list(model.cfg.model.data_preprocessor.size),
        "threshold": model.decode_head.threshold,
        "checkpoint": str(checkpoint.resolve()),
        "foreground": lesion,
        "background": background,
        "mean_iou": (lesion["iou"] + background["iou"]) / 2,
        "mean_dice_f1": (lesion["dice"] + background["dice"]) / 2,
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
    }
    write_csv(output / "per_image.csv", rows)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ranked = sorted(rows, key=lambda row: (row["dice"], row["image_id"]))
    for name, row in zip(("poor", "median", "strong"), (ranked[0], ranked[len(rows) // 2], ranked[-1])):
        image_id = row["image_id"]
        pair = manifest[image_id]
        plot_case(read_image(data / pair["image_path"]), read_mask(data / pair["mask_path"]),
                  read_mask(output / row["prediction"]), output / f"{name}_{image_id}.png")

    review_ids = sorted(random.Random(42).sample(sorted(test_ids), min(40, len(test_ids))))
    areas = {row["image_id"]: row["area_fraction"] for row in rows}
    write_csv(output / "review.csv", [
        dict(image_id=image_id,
             image_path=str((data / manifest[image_id]["image_path"]).resolve()),
             reference_path=str((data / manifest[image_id]["mask_path"]).resolve()),
             small_lesion=int(areas[image_id] < 0.05),
             hair_crossing="", faint_border="", jagged_border="")
        for image_id in review_ids
    ])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the frozen model on the prepared held-out set.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint, args.data, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
