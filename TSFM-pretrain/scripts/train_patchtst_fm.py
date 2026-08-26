from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import yaml
from torch.multiprocessing import set_sharing_strategy, set_start_method
from torch.utils.data import DataLoader

from tsfm.data.dataset import MixtureDataset
from tsfm.model.patchtst_fm import (
    PatchTSTFMArrowDataset,
    PatchTSTFMConfig,
    PatchTSTFMForPretraining,
    PatchTSTFMGiftEval,
    PatchTSTFMPatternRandomWalkDataset,
)
from tsfm.train.trainer import Trainer, worker_init_fn


def main():
    parser = argparse.ArgumentParser(description="Pre-train PatchTST-FM.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patchtst_fm.yaml",
        help="Path to YAML config file.",
    )

    # Mutually exclusive data sources
    parser.add_argument(
        "--arrow_path",
        type=str,
        default="kernelsynth-data.arrow",
        help="Path to a KernelSynth Arrow IPC file.",
    )
    parser.add_argument(
        "--gifteval_root",
        type=str,
        default="/data/parietal/store4/data/GiftEvalPretrain_nostream",
        help="Path to locally-downloaded GiftEvalPretrain directory.",
    )
    parser.add_argument(
        "--tsmixup_root",
        type=str,
        default="/data/parietal/store4/data/tsmixup_nostream",
        help="Path to locally-downloaded tsmixup directory.",
    )
    parser.add_argument(
        "--val_arrow_path",
        type=str,
        default=None,
        help="Optional path to a separate Arrow file for validation. If not provided, the training "
        "data will be used for validation",
    )
    # Compatibility with launchers that inject local rank as a CLI arg.
    parser.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    rank = int(os.environ.get("RANK", 0))
    is_master = rank == 0

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model_cfg = PatchTSTFMConfig(**config.model)
    model = PatchTSTFMForPretraining(model_cfg)

    if is_master:
        print(model.backbone.model_summary())

    # ------------------------------------------------------------------ #
    # Build datasets
    # ------------------------------------------------------------------ #
    train_datasets = []
    # ── KernelSynth Arrow source ───────────────────────────────────
    if config.mixture["use_mixture"]:
        if is_master:
            print(f"Opening Arrow file: {args.arrow_path}")

        dummy_root = "dummy_root_for_arrow"  # PatchTSTFMGiftEval requires a root, but it won't be used when arrow_paths is provided
        os.makedirs(dummy_root, exist_ok=True)

        train_dataset = PatchTSTFMGiftEval(
            root=dummy_root,
            arrow_paths=[args.arrow_path],
            batch_size=config.micro_batch_size,
            **config.data,
        )

        train_datasets.append(train_dataset)

    # ── TSFMixup Arrow root ───────────────────────────────────
    if config.mixture["use_mixture"]:
        if is_master:
            print(f"Opening TSMixup at: {args.tsmixup_root}")

        train_dataset = PatchTSTFMGiftEval(
            root=dummy_root,
            arrow_paths=[args.tsmixup_root],
            batch_size=config.micro_batch_size,
            **config.data,
        )

        #train_dataset = PatchTSTFMGiftEval(
        #    root=args.tsmixup_root,
        #    batch_size=config.micro_batch_size,
        #    **config.data,
        #)

        train_datasets.append(train_dataset)

    # ── GiftEvalPretrain local source ──────────────────────────────
    if is_master:
        print(f"Opening GiftEvalPretrain at: {args.gifteval_root}")

    gifteval_cfg = getattr(config, "gifteval", {})

    train_dataset = PatchTSTFMGiftEval(
        root=args.gifteval_root,
        batch_size=config.micro_batch_size,
        num_files_per_batch=config.files_per_micro_batch,
        **{**config.data, **gifteval_cfg},
    )

    train_datasets.append(train_dataset)

    # ── On-the-fly synthetic pattern random-walk source ───────────────────
    pattern_cfg = getattr(config, "pattern_rw", {}) or {}
    has_pattern = bool(pattern_cfg.get("use", False))
    if has_pattern:
        if is_master:
            print("Enabling on-the-fly PatternRandomWalk synthetic dataset")

        pattern_dataset = PatchTSTFMPatternRandomWalkDataset(
            context_length=config.data["context_length"],
            d_patch=config.data["d_patch"],
            batch_size=config.micro_batch_size,
            mask_ratio=config.data.get("mask_ratio", 0.4),
            n_cpm=config.data.get("n_cpm", 8),
            min_past=config.data.get("min_past", 1),
            random_context_length=config.data.get("random_context_length", True),
            **{k: v for k, v in pattern_cfg.items() if k not in {"use", "weight"}},
        )
        train_datasets.append(pattern_dataset)

    # ── Combine with MixtureDataset if both sources are used ───────────────
    if len(train_datasets) > 1:
        # ── Compute mixture weights ────────────────────────────────────────
        has_ks = config.mixture["use_mixture"] and os.path.exists(args.arrow_path)
        has_ts = config.mixture["use_mixture"] #and os.path.isdir(args.tsmixup_root)

        raw_weights: list[float] = []
        if has_ks:
            raw_weights.append(1.0)  # KernelSynth share of synthetic 1.0
        if has_ts:
            raw_weights.append(1.0)  # TSMixup share of synthetic 1.0
        raw_weights.append(3.0)        # GiftEval
        if has_pattern:
            raw_weights.append(float(pattern_cfg.get("weight", 1.0)))

        total = sum(raw_weights)
        mixture_weights = [w / total for w in raw_weights]

        if is_master:
            labels = (
                (["KernelSynth"] if has_ks else [])
                + (["TSMixup"] if has_ts else [])
                + ["GiftEval"]
                + (["PatternRW"] if has_pattern else [])
            )
            for label, w in zip(labels, mixture_weights):
                print(f"  Mixture weight  {label}: {w:.4f}")

        train_dataset = MixtureDataset(
            train_datasets,
            strategy="given",
            weights=mixture_weights,
        )
    else:
        # Single source — wrap so the rest of the code stays uniform.
        train_dataset = train_datasets[0]

    if is_master:
        print(f"Loaded {len(train_dataset)} Time Series.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory="cuda" in config.device,
        persistent_workers=True,
        num_workers=8,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
    )

    # Validation dataset (if provided)
    if args.val_arrow_path is not None:
        val_dataset = PatchTSTFMArrowDataset(
            arrow_path=args.val_arrow_path,
            mode="validation",
            batch_size=config.micro_batch_size,
            **config.data,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            pin_memory="cuda" in config.device,
            persistent_workers=True,
            num_workers=2,
            prefetch_factor=2,
        )
    else:
        val_dataloader = None

    if is_master:
        print(
            f"Per-GPU batch size: {config.micro_batch_size * config.gradient_accumulation_steps}"
            f" Micro-batch: {config.micro_batch_size}"
        )

    trainer = Trainer(config, model, train_dataloader, val_dataloader=val_dataloader)
    trainer.train()


if __name__ == "__main__":
    set_sharing_strategy("file_system")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
