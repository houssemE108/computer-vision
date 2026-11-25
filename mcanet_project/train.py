import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .inference import load_model, predict, save_prediction
from .masks import pixel_counts, read_image, read_mask, scores
from .prepare_data import load_manifest, read_split, validate_splits
from .setup_upstream import COMMIT, UPSTREAM_CONFIG


WEIGHTS = Path("weights/mscan_t_20230227-119e8c9f.pth")


def configure(args, cfg, work_dir, split_file, image_dir, num_train, on_cuda):
    size = (args.size, args.size)
    cfg.model.data_preprocessor.size = size
    cfg.model.decode_head.image_size = args.size // 4
    cfg.model.decode_head.threshold = 0.3
    if args.random_init:
        cfg.model.backbone.init_cfg = None
    else:
        cfg.model.backbone.init_cfg.checkpoint = str(args.pretrained.resolve())

    resize = dict(type="Resize", scale=size, keep_ratio=False)
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile", ignore_empty=False, color_type="color_ignore_orientation"),
        dict(type="LoadAnnotations"),
        resize,
    ]
    if not args.smoke:
        cfg.train_pipeline += [
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
        ]
    cfg.train_pipeline += [dict(type="PackSegInputs")]
    cfg.test_pipeline = [
        dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
        resize,
        dict(type="LoadAnnotations"),
        dict(type="PackSegInputs"),
    ]

    def loader(ann_file, pipeline, training=False):
        return dict(
            batch_size=args.batch_size if training else 1,
            num_workers=0,
            sampler=dict(type="InfiniteSampler" if training else "DefaultSampler", shuffle=training),
            dataset=dict(
                type="MedicalUniversalDataset",
                data_root=str(args.data.resolve()),
                data_prefix=dict(img_path=str(image_dir), seg_map_path="masks"),
                ann_file=str(ann_file.resolve()),
                img_suffix=".jpg",
                seg_map_suffix=".png",
                pipeline=pipeline,
            ),
        )

    cfg.train_dataloader = loader(split_file, cfg.train_pipeline, training=True)
    quick_run = args.smoke or args.pilot
    steps_per_epoch = math.ceil(num_train / args.batch_size)
    steps = (args.steps or (40 if args.smoke else 100)) if quick_run else steps_per_epoch * args.epochs
    cfg.train_cfg.max_iters = steps
    cfg.train_cfg.val_interval = steps if quick_run else steps_per_epoch
    if quick_run:
        cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
        cfg.test_dataloader = cfg.test_cfg = cfg.test_evaluator = None
    else:
        cfg.val_dataloader = loader(args.data / "val.txt", cfg.test_pipeline)
        cfg.test_dataloader = loader(args.data / "test.txt", cfg.test_pipeline)
    decay = cfg.param_scheduler[-1]
    decay.end = steps
    warmup = min(1500, max(2, round(steps * 1500 / 80000)))
    if quick_run or steps <= warmup + 1:
        decay.begin = 0
        cfg.param_scheduler = [decay]
    else:
        cfg.param_scheduler[0].end = warmup
        decay.begin = warmup
    if on_cuda and not args.no_amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
    cfg.env_cfg.mp_cfg.mp_start_method = "spawn"
    cfg.env_cfg.dist_cfg.backend = "gloo"
    cfg.env_cfg.cudnn_benchmark = False
    cfg.randomness = dict(seed=42)
    cfg.work_dir = str(work_dir)
    cfg.default_hooks.logger.interval = 1 if quick_run else min(50, steps_per_epoch)
    cfg.default_hooks.checkpoint.update(
        interval=cfg.train_cfg.val_interval, max_keep_ckpts=1, save_last=True
    )
    if not quick_run:
        cfg.default_hooks.checkpoint.update(save_best="mDice", rule="greater")
    return steps


def save_learning_curves(runner, output, steps_per_epoch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    loss = runner.message_hub.get_scalar("train/loss").data[0]
    dice = runner.message_hub.get_scalar("val/mDice").data[0]
    figure = Figure(figsize=(8, 3), layout="constrained")
    FigureCanvasAgg(figure)
    train_plot, val_plot = figure.subplots(1, 2)
    train_plot.plot(np.arange(1, len(loss) + 1), loss)
    train_plot.set(xlabel="Training step", ylabel="Training loss")
    val_plot.plot(np.arange(1, len(dice) + 1) * steps_per_epoch, dice, marker="o")
    val_plot.set(xlabel="Training step", ylabel="Validation mean Dice (%)")
    figure.savefig(output, dpi=150)


def logged_peak_memory(log_dir):
    # MMEngine reads and resets the CUDA peak each time it logs memory.
    scalars = Path(log_dir) / "vis_data" / "scalars.json"
    with scalars.open(encoding="utf-8") as file:
        return max((json.loads(line).get("memory", 0) for line in file), default=0)


def main():
    parser = argparse.ArgumentParser(description="Train the pinned MCANet-T on prepared ISIC pairs.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, help="Iteration count for a smoke test or pilot.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained", type=Path, default=WEIGHTS)
    initialization.add_argument("--random-init", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1 or (args.steps is not None and args.steps < 2):
        parser.error("Use positive batch size and epochs, and at least two smoke/pilot steps.")
    if args.steps is not None and not (args.pilot or args.smoke):
        parser.error("--steps applies only to --smoke or --pilot.")
    if not UPSTREAM_CONFIG.is_file():
        parser.error("Install the pinned upstream source first. See README.md.")
    if not args.random_init and (not args.pretrained.is_file() or not args.pretrained.stat().st_size):
        parser.error(f"Missing {args.pretrained}. Supply the official weights or use --random-init.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Choose an empty run directory.")

    manifest = load_manifest(args.data)
    split_file = args.data / "train.txt"
    train_ids = read_split(split_file)
    if args.smoke:
        train_ids = train_ids[:4]
        if len(train_ids) != 4:
            parser.error("The smoke test needs four training pairs.")
    if not train_ids or any(image_id not in manifest for image_id in train_ids):
        parser.error("Training list is empty or contains unknown IDs.")
    if not (args.smoke or args.pilot):
        try:
            validate_splits(args.data, manifest)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.utils import register_all_modules

    on_cuda = torch.cuda.is_available()
    if not on_cuda and not args.smoke:
        parser.error("CUDA is required for the pilot and full run. CPU can run --smoke.")
    if not on_cuda:
        torch.set_num_threads(4)
    register_all_modules()
    work_dir = args.run_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        split_file = work_dir / "smoke_train.txt"
        split_file.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    example_path = (args.data / manifest[train_ids[0]]["image_path"]).resolve()
    cfg = Config.fromfile(UPSTREAM_CONFIG)
    steps = configure(args, cfg, work_dir, split_file, example_path.parent, len(train_ids), on_cuda)
    config_path = work_dir / "resolved_config.py"
    cfg.dump(str(config_path))
    run_settings = {
        "upstream_commit": COMMIT,
        "mode": "smoke" if args.smoke else ("pilot" if args.pilot else "train"),
        "device": torch.cuda.get_device_name() if on_cuda else "CPU",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "input_size": args.size,
        "batch_size": args.batch_size,
        "steps": steps,
        "initialization": "random" if args.random_init else str(args.pretrained.resolve()),
    }
    settings_path = work_dir / "run.json"
    settings_path.write_text(json.dumps(run_settings, indent=2) + "\n", encoding="utf-8")
    runner = Runner.from_cfg(cfg)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = runner.train()
    run_settings["training_seconds"] = time.perf_counter() - start
    if on_cuda:
        run_settings["peak_allocated_mib"] = max(
            logged_peak_memory(runner.log_dir), torch.cuda.max_memory_allocated() // 2**20
        )

    if not (args.smoke or args.pilot):
        save_learning_curves(runner, work_dir / "learning_curves.png",
                             math.ceil(len(train_ids) / args.batch_size))
    if args.smoke:
        losses = runner.message_hub.get_scalar("train/loss").data[0].tolist()
        if not all(math.isfinite(loss) for loss in losses):
            raise RuntimeError("The smoke test produced a non-finite loss.")
        image = read_image(example_path)
        model.cfg = Config.fromfile(config_path)
        before = predict(model.eval(), image)
        checkpoint = work_dir / f"iter_{steps}.pth"
        del model, runner
        reloaded = load_model(config_path, checkpoint, "cuda:0" if on_cuda else "cpu")
        after = predict(reloaded, image)
        if not np.array_equal(before, after):
            raise RuntimeError("Predictions differ after checkpoint reload.")
        save_prediction(image, after, work_dir / "prediction")
        reference = read_mask(args.data / manifest[train_ids[0]]["mask_path"])
        run_settings.update(
            first_loss=losses[0], final_loss=losses[-1], example_id=train_ids[0],
            checkpoint_reload_matches=True, foreground_pixels=int(after.sum()),
            training_example_dice=scores(pixel_counts(after, reference))["dice"],
        )
    settings_path.write_text(json.dumps(run_settings, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_settings, indent=2))


if __name__ == "__main__":
    main()
