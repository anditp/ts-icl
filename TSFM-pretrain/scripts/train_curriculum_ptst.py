from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import yaml
from torch.multiprocessing import set_sharing_strategy, set_start_method
from torch.utils.data import DataLoader

from tsfm.data.dataset import MixtureDataset
from tsfm.model.patchtst_fm import (
    PatchTSTFMConfig,
    PatchTSTFMForPretraining,
    PatchTSTFMGiftEval,
)
from tsfm.train.trainer import Trainer, worker_init_fn

def train_stage(config, model, dataset, max_steps, resume=True):
    """Orchestrates a single training stage with a specific dataset."""
    config.max_steps = max_steps
    config.resume_training = resume

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory="cuda" in config.device,
        persistent_workers=True,
        num_workers=8,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
    )

    # unwrap an existing DDP-wrapped model if necessary
    base_model = model.module if hasattr(model, "module") else model

    trainer = Trainer(config, base_model, dataloader)
    trainer.train()

    # get the raw underlying nn.Module back (Trainer keeps a raw_model attribute)
    new_model = getattr(trainer, "raw_model", None)
    if new_model is None:
        # fallback: Trainer.model may be DDP, so try model.module
        tm = getattr(trainer, "model", None)
        if hasattr(tm, "module"):
            new_model = tm.module
        else:
            new_model = tm or base_model

    # return an unwrapped module for next stage
    return new_model

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
        "--use_mixture",
        default=True,
        help="Whether to combine the Arrow and GiftEval datasets using a MixtureDataset."
        " If false, only GiftEval will be used",
    )
    parser.add_argument(
        "--val_arrow_path",
        type=str,
        default=None,
        help="Optional path to a separate Arrow file for validation. If not provided, the training "
        "data will be used for validation",
    )
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
    if args.use_mixture:
        if is_master:
            print(f"Opening Arrow file: {args.arrow_path}")
        
        dummy_root = "dummy_root_for_arrow"  # PatchTSTFMGiftEval requires a root, but it won't be used when arrow_paths is provided
        os.makedirs(dummy_root, exist_ok=True)

        kernelsynth_dataset = PatchTSTFMGiftEval(
            root=dummy_root,
            arrow_paths=[args.arrow_path],
            batch_size=config.micro_batch_size,
            **config.data,
        )

        train_datasets.append(kernelsynth_dataset)

    # ── TSFMixup Arrow root ───────────────────────────────────
    if args.use_mixture:
        if is_master:
            print(f"Opening TSMixup at: {args.tsmixup_root}")
        

        tsmixup_dataset = PatchTSTFMGiftEval(
            root=args.tsmixup_root,
            batch_size=config.micro_batch_size,
            **config.data,
        )

        train_datasets.append(tsmixup_dataset)

    # ── GiftEvalPretrain local source ──────────────────────────────
    if is_master:
        print(f"Opening GiftEvalPretrain at: {args.gifteval_root}")

    gifteval_cfg = getattr(config, "gifteval", {})

    gifteval_dataset = PatchTSTFMGiftEval(
        root=args.gifteval_root,
        batch_size=config.micro_batch_size,
        **{**config.data, **gifteval_cfg},
    )

    train_datasets = [kernelsynth_dataset, tsmixup_dataset, gifteval_dataset]
    stage2_datasets = [kernelsynth_dataset, tsmixup_dataset]

    # ── Combine with MixtureDataset if both sources are used ───────────────
    train_dataset = MixtureDataset(
        train_datasets,
        strategy="given",
        weights=[0.1, 0.4, 0.5]
    )
    
    stage2_dataset = MixtureDataset(
        stage2_datasets,
        strategy="given",
        weights=[0.2, 0.8]
    )

    if is_master:
        print(f"Loaded {len(train_dataset)} Time Series.")

    if is_master:
        print(
            f"Per-GPU batch size: {config.micro_batch_size * config.gradient_accumulation_steps}"
            f" Micro-batch: {config.micro_batch_size}"
        )
    
    # Training stages
    stages = [
        (train_dataset, 100000),
    ]

    for i, (data_path, end_step) in enumerate(stages):
        if is_master:
            print(f"\n--- STARTING STAGE {i}: {data_path} (Up to step {end_step}) ---")
        model = train_stage(config, model, data_path, end_step, resume=True)


if __name__ == "__main__":
    set_sharing_strategy("file_system")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
