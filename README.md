# MCANet-T skin lesion segmentation

A university reproduction project for ISIC 2018 Task 1. The authors’ **MSCAN-T encoder** extracts features at four scales. The **MCA decoder** fuses the three deeper features using strip convolutions and cross-axis attention, then adds shallow features for boundary detail. MMSegmentation resizes the final scores to the original image dimensions and uses a sigmoid threshold of 0.3 to produce a binary mask. The local demo displays that mask and its contour for educational use.

## Try the checked example in this Windows workspace

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 -- --config runs/demo-smoke/resolved_config.py --checkpoint runs/demo-smoke/iter_480.pth --device cpu
```

Select `ISIC_0000198` and click **Segment**. The checkpoint was trained on four labeled pairs as a smoke test. See [the comparison panel](runs/demo-smoke/prediction/comparison.png) and [run details](runs/demo-smoke/run.json).

## Set up once

Run the commands from this project folder using 64-bit Python 3.10. Ubuntu 22.04 is the upstream-recommended route. The setup script also handles Windows-compatible upstream config filenames. For GPU training, use an NVIDIA driver that supports CUDA 11.8 and have at least 30 GB free if keeping both ISIC ZIP files and the extracted data.

```sh
python -m venv .venv
# Ubuntu or WSL
source .venv/bin/activate
# Windows PowerShell equivalent
# .\.venv\Scripts\Activate.ps1

curl -fL https://codeload.github.com/haoshao-nku/medical_seg/zip/9a18e129f6f7db3018b509677ef426c67a616e0b -o medical_seg.zip
python -m mcanet_project.setup_upstream medical_seg.zip
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=mmcv mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html -r requirements.txt -e upstream/mmsegmentation
python -m pip check
python -c "import torch, mmcv.ops; from mmseg.utils import register_all_modules; register_all_modules(); print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Use `curl.exe` in PowerShell. The setup script only renames asterisks in archive filenames. For CPU smoke tests and demo inference, substitute `/whl/cpu` and `/dist/cpu/` in the two wheel URLs and select `--device cpu` for inference. The training command uses CUDA when available. If you created the environment with `uv venv`, run `python -m ensurepip` because the upstream MCA decoder imports `pip`.

Download the [configured MSCAN-T encoder checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth) and save it to `weights/mscan_t_20230227-119e8c9f.pth`. Use `--random-init` explicitly if pretraining is unavailable. Load checkpoints from sources you trust.

## Prepare the data and run the smoke test

Download **2018 → Task 1 → Training Data and Training Ground Truth** from the [ISIC data portal](https://challenge.isic-archive.com/data/#2018). Extract the JPEGs into `data/images` and the matching mask PNGs into `data/masks`. The directories may contain the extra wrapper folder supplied in the ZIP. Use only the labeled training archives for this experiment.

```sh
python -m mcanet_project.prepare_data --images data/images --masks data/masks --output data/isic
python -m mcanet_project.train --data data/isic --run-dir runs/smoke --smoke --size 256
```

Preparation validates all 2,594 pairs, creates 0/1 masks, groups exact image duplicates, and freezes seed-42 splits targeting 1,867/207/520. Pass `--groups-csv path.csv` with `image_id,group_id` when trusted patient or lesion IDs are available. Pass `--author-splits path` if you have verified author `train.txt` and `test.txt` files, one ISIC ID per line. No author split lists were found in the pinned source.

The smoke run uses four training pairs, saves and reloads a checkpoint, compares predictions, and writes `runs/smoke/prediction/mask.png` and `overlay.png`. With only four isolated pairs, add `--smoke` to data preparation and use a separate output folder. This workspace includes the prepared four-pair sample in `data/smoke-isic`. For a full run from this workspace, add `--training-ids data/smoke-isic/train.txt` to data preparation to keep those pairs and their groups in training. Smoke losses and overlaps measure a training example only.

## Train, evaluate, and demo

```sh
python -m mcanet_project.train --data data/isic --run-dir runs/pilot --pilot --size 512
# Use a fresh run directory and --size 256 if 512 does not fit in memory.
python -m mcanet_project.train --data data/isic --run-dir runs/final --epochs 1 --size 512
python -m mcanet_project.evaluate --data data/isic --config runs/final/resolved_config.py --checkpoint runs/final/BEST_CHECKPOINT.pth --output runs/test
streamlit run app.py --server.address 127.0.0.1 -- --config runs/final/resolved_config.py --checkpoint runs/final/BEST_CHECKPOINT.pth
```

Replace `BEST_CHECKPOINT.pth` with the actual `best_mDice_iter_*.pth` filename. Keep the pilot's working resolution for training and budget up to ten epochs based on the measured step time within your compute limit. The default training batch is 1. CUDA uses mixed precision, and the best checkpoint is chosen by validation mean Dice. The run folder contains the resolved config, checkpoint, loss and validation curves, and device details. Put up to three permitted demo JPEGs in `data/examples`, or upload an image directly. An optional same-size binary PNG enables overlap scores.

If MMEngine's environment check raises a `distutils` assertion on Windows with Python 3.10, set `$env:SETUPTOOLS_USE_DISTUTILS = 'stdlib'` in PowerShell before running training.

Test results include pooled lesion IoU, Dice, precision and recall, two-class mean IoU and Dice, and warmed-up batch-one latency including model preprocessing and mask decoding. The evaluator writes `metrics.json`, `per_image.csv`, representative panels, and a fixed 40-image `review.csv`. Fill the review flags from the original and the reference mask before looking at predictions. Mark hair crossing the lesion, a visibly faint border over half its length, and a visibly jagged or lobulated border. Small lesions occupy less than 5% of the image. Run `python -m mcanet_project.review runs/test` to summarize the annotated groups.

## Reference and adaptations

The checked-in tiny ISIC config uses sigmoid cross-entropy and AdamW with learning rate 0.00006, weight decay 0.01, and a tenfold head learning-rate multiplier. This project keeps those code settings, uses fixed square input resizing and a shorter schedule, and reserves validation data for checkpoint selection. The paper reports a different batch size, optimizer configuration, and budget. Its Table 2 reports 90.40% mIoU and 92.93% F1 for MCANet-T. Compare the scores only with the split and metric definitions stated. [Shao et al., 2025](https://doi.org/10.1007/s11633-025-1552-6) · [Authors’ code](https://github.com/haoshao-nku/medical_seg) · [ISIC data and attribution](https://challenge.isic-archive.com/data/#2018)
