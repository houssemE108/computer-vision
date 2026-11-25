import numpy as np
from PIL import Image, ImageFilter


def read_image(source):
    with Image.open(source) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Choose a JPEG or PNG image.")
        return np.asarray(image.convert("RGB"))


def read_mask(source):
    with Image.open(source) as image:
        if image.format != "PNG" or image.mode not in {"1", "L", "P"}:
            raise ValueError("Choose a single-channel binary PNG mask.")
        mask = np.asarray(image)
    values = set(np.unique(mask))
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("Mask values must use either 0/1 or 0/255.")
    return mask > 0


def save_mask(mask, path):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def contour_overlay(image, mask):
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same dimensions.")
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    edge = Image.fromarray((mask & ~interior).astype(np.uint8) * 255)
    line_width = max(3, 2 * round(min(mask.shape) / 800) + 1)
    result = image.copy()
    result[np.asarray(edge.filter(ImageFilter.MaxFilter(line_width))) > 0] = [0, 255, 100]
    return result


def pixel_counts(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError("Reference mask must match the image dimensions.")
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)
    return {
        "tp": int(np.count_nonzero(prediction & reference)),
        "fp": int(np.count_nonzero(prediction & ~reference)),
        "fn": int(np.count_nonzero(~prediction & reference)),
        "tn": int(np.count_nonzero(~prediction & ~reference)),
    }


def scores(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (union + tp) if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
