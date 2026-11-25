import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .masks import contour_overlay, read_image, save_mask


def load_model(config, checkpoint, device):
    for path in (config, checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    import torch
    from mmseg.apis import init_model
    from mmseg.utils import register_all_modules

    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. For inference, retry with --device cpu.")
    register_all_modules()
    return init_model(str(config), str(checkpoint), device=device)


def predict(model, image):
    from mmseg.apis import inference_model

    bgr = np.ascontiguousarray(image[:, :, ::-1])
    prediction = inference_model(model, bgr).pred_sem_seg.data[0].cpu().numpy()
    if prediction.shape != image.shape[:2] or not np.isin(prediction, (0, 1)).all():
        raise RuntimeError("The model did not return an original-size binary mask.")
    return prediction.astype(bool)


def save_prediction(image, mask, output):
    output.mkdir(parents=True, exist_ok=True)
    save_mask(mask, output / "mask.png")
    Image.fromarray(contour_overlay(image, mask)).save(output / "overlay.png")


def main():
    parser = argparse.ArgumentParser(description="Save the mask and contour for one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    image = read_image(args.image)
    model = load_model(args.config, args.checkpoint, args.device)
    save_prediction(image, predict(model, image), args.output)
    print(f"Saved mask.png and overlay.png in {args.output}.")


if __name__ == "__main__":
    main()
