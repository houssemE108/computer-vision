import argparse
import csv
from pathlib import Path
from statistics import mean

from .evaluate import write_csv


def main():
    parser = argparse.ArgumentParser(description="Summarize Dice for the annotated error-analysis sample.")
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    with (args.evaluation_dir / "per_image.csv").open(encoding="utf-8", newline="") as file:
        dice = {row["image_id"]: float(row["dice"]) for row in csv.DictReader(file)}
    with (args.evaluation_dir / "review.csv").open(encoding="utf-8", newline="") as file:
        sample = list(csv.DictReader(file))
    results = []
    for feature in ("small_lesion", "hair_crossing", "faint_border", "jagged_border"):
        for present in ("0", "1"):
            if any(row[feature] not in {"0", "1"} for row in sample):
                parser.error(f"Fill every {feature} value in review.csv with 0 or 1.")
            values = [dice[row["image_id"]] for row in sample if row[feature] == present]
            results.append(dict(feature=feature, present=present, count=len(values),
                                mean_lesion_dice=mean(values) if values else None))
    output = args.evaluation_dir / "review_summary.csv"
    write_csv(output, results)
    print(f"Saved {output}.")


if __name__ == "__main__":
    main()
