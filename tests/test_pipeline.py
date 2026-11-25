import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from mcanet_project.evaluate import evaluate
from mcanet_project.masks import contour_overlay, pixel_counts, read_image, read_mask, scores
from mcanet_project.prepare_data import group_images, load_manifest, make_splits, prepare, read_split


class MaskTests(unittest.TestCase):
    def test_known_scores(self):
        prediction = np.array([[1, 1], [0, 0]])
        reference = np.array([[1, 0], [1, 0]])
        self.assertEqual(pixel_counts(prediction, reference), dict(tp=1, fp=1, fn=1, tn=1))
        self.assertEqual(scores(pixel_counts(prediction, reference)),
                         dict(iou=1 / 3, dice=0.5, precision=0.5, recall=0.5))
        self.assertEqual(scores(pixel_counts(reference, reference))["dice"], 1)
        self.assertEqual(scores(pixel_counts(1 - reference, reference))["dice"], 0)

    def test_empty_masks_and_wrong_dimensions(self):
        empty = np.zeros((3, 5), dtype=bool)
        result = scores(pixel_counts(empty, empty))
        self.assertEqual(result, dict(iou=1.0, dice=1.0, precision=None, recall=None))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            pixel_counts(empty, empty.T)

    def test_contour_matches_original_coordinates(self):
        image = np.zeros((15, 23, 3), dtype=np.uint8)
        mask = np.zeros((15, 23), dtype=bool)
        mask[3:12, 6:17] = True
        overlay = contour_overlay(image, mask)
        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue((overlay[3, 11] == [0, 255, 100]).all())
        self.assertFalse(overlay[7, 11].any())
        self.assertFalse(overlay[0, 0].any())
        self.assertTrue(np.array_equal(contour_overlay(image, np.zeros_like(mask)), image))

    def test_mask_rejects_intermediate_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.array([[0, 65, 255]], dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "0/255"):
                read_mask(path)


class DatasetTests(unittest.TestCase):
    def test_prepares_four_pairs_and_freezes_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks, output = root / "images", root / "masks", root / "prepared"
            images.mkdir()
            masks.mkdir()
            for index in range(4):
                image_id = f"ISIC_{index:07d}"
                image = np.full((12, 20, 3), 10 + 30 * index, dtype=np.uint8)
                mask = np.zeros((12, 20), dtype=np.uint8)
                mask[3:8, 4:13] = 255
                Image.fromarray(image).save(images / f"{image_id}.jpg")
                Image.fromarray(mask).save(masks / f"{image_id}_segmentation.png")

            first = prepare(images, masks, output, smoke=True)
            ids = read_split(output / "train.txt")
            self.assertEqual(first["counts"], {"train": 4})
            self.assertFalse((output / "test.txt").exists())
            self.assertEqual(prepare(images, masks, output, smoke=True), first)
            self.assertEqual(read_split(output / "train.txt"), ids)
            row = load_manifest(output)[ids[0]]
            self.assertEqual(read_image(output / row["image_path"]).shape, (12, 20, 3))
            with Image.open(output / row["mask_path"]) as normalized:
                self.assertEqual(set(np.unique(normalized)), {0, 1})
            (masks / f"{ids[0]}_segmentation.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing masks"):
                prepare(images, masks, output, smoke=True)

    def test_grouped_split_is_complete_disjoint_and_repeatable(self):
        ids = [f"ISIC_{index:07d}" for index in range(2594)]
        fingerprints = {image_id: index for index, image_id in enumerate(ids)}
        fingerprints[ids[1]] = fingerprints[ids[0]]
        groups = group_images(fingerprints, {ids[1]: "lesion-a", ids[2]: "lesion-a"})
        self.assertEqual(len({groups[image_id] for image_id in ids[:3]}), 1)
        splits = make_splits(groups)
        self.assertEqual(splits, make_splits(groups))
        self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [1867, 207, 520])
        self.assertEqual(sorted(sum(splits.values(), [])), ids)
        self.assertEqual(sum(len({groups[image_id] for image_id in part}) for part in splits.values()),
                         len(set(groups.values())))
        reserved = [*splits["test"][:3], ids[0]]
        training = set(make_splits(groups, training_ids=reserved)["train"])
        self.assertTrue(set(reserved + ids[:3]) <= training)


class EvaluationTests(unittest.TestCase):
    def test_report_writes_scores_and_panels_for_a_known_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.zeros((16, 24), dtype=bool)
            reference[4:12, 8:16] = True
            Image.fromarray(reference).save(root / "mask.png")
            Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8)).save(root / "image.jpg")
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["image_id", "image_path", "mask_path", "group_id"])
                for split in ("train", "val", "test"):
                    writer.writerow([split, "image.jpg", "mask.png", split])
                    (root / f"{split}.txt").write_text(split + "\n", encoding="utf-8")

            preprocessor = SimpleNamespace(size=(256, 256))
            model = SimpleNamespace(
                cfg=SimpleNamespace(model=SimpleNamespace(data_preprocessor=preprocessor)),
                decode_head=SimpleNamespace(threshold=0.3),
            )
            with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
                with patch("mcanet_project.evaluate.load_model", return_value=model):
                    with patch("mcanet_project.evaluate.predict", return_value=reference):
                        report = evaluate(root / "config.py", root / "model.pth", root,
                                          root / "results", "cpu")
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["foreground"]["dice"], 1)
            self.assertEqual(report["mean_iou"], 1)
            self.assertTrue((root / "results/median_test.png").is_file())
            self.assertTrue((root / "results/review.csv").is_file())


if __name__ == "__main__":
    unittest.main()
