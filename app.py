import argparse
import io
from pathlib import Path

import streamlit as st

from mcanet_project.inference import load_model, predict
from mcanet_project.masks import contour_overlay, pixel_counts, read_image, read_mask, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--examples", type=Path, default=Path("data/examples"))
    return parser.parse_args()


@st.cache_resource
def cached_model(config, checkpoint, device, checkpoint_modified):
    return load_model(config, checkpoint, device)


def main():
    args = parse_args()
    st.set_page_config(page_title="Lesion segmentation", layout="wide")
    st.title("Lesion segmentation")
    st.caption("MCANet-T university project for educational use.")

    examples = {path.stem: path for path in sorted(args.examples.glob("*.jpg"))[:3]}
    selection = None
    if examples:
        example = st.selectbox("Example image", list(examples), index=None,
                               placeholder="Choose an example")
        selection = examples.get(example)
    upload = st.file_uploader("Dermoscopic image", type=["jpg", "jpeg", "png"])
    reference_file = st.file_uploader("Reference mask (optional)", type=["png"])
    if not st.button("Segment", type="primary", disabled=upload is None and selection is None):
        return

    try:
        image = read_image(io.BytesIO(upload.getvalue()) if upload else selection)
        reference = read_mask(io.BytesIO(reference_file.getvalue())) if reference_file else None
        if reference is not None and reference.shape != image.shape[:2]:
            raise ValueError("Reference mask must match the image dimensions.")
        with st.spinner("Segmenting image…"):
            model = cached_model(args.config, args.checkpoint, args.device,
                                 args.checkpoint.stat().st_mtime_ns)
            mask = predict(model, image)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        st.error(str(error))
        return

    original, predicted, overlay = st.columns(3)
    original.image(image, caption="Original")
    predicted.image(mask.astype("uint8") * 255, caption="Predicted mask")
    overlay.image(contour_overlay(image, mask), caption="Lesion contour")
    if reference is not None:
        result = scores(pixel_counts(mask, reference))
        labels = {"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall"}
        for column, (name, value) in zip(st.columns(4), result.items()):
            column.metric(labels[name], f"{value:.1%}" if value is not None else "N/A")


if __name__ == "__main__":
    main()
