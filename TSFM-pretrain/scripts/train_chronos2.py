import argparse
import os
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import yaml
from torch.multiprocessing import set_sharing_strategy, set_start_method
from torch.utils.data import DataLoader

from tsfm.model.chronos2 import Chronos2CoreConfig, Chronos2Dataset, Chronos2Model
from tsfm.train.trainer import Trainer, worker_init_fn


def main():
    parser = argparse.ArgumentParser(description="Train a TSFM model.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    rank = int(os.environ.get("RANK", 0))
    is_master = rank == 0

    # Model config
    chronos_config = Chronos2CoreConfig(**config.model)
    model = Chronos2Model(chronos_config).to(config.device)

    if is_master:
        print("Opening dataset and preparing dataloaders...")

    # Memory-mapped Arrow loading: the file is mapped into virtual memory
    # without reading it into RAM. The OS pages in only the bytes accessed
    # during training. Startup is near-instant regardless of file size.
    arrow_path = "./kernelsynth-data.arrow"
    mmap_source = pa.memory_map(arrow_path, "r")
    table = pa.ipc.open_file(mmap_source).read_all()
    n_samples = len(table)
    if is_master:
        print(
            f"Memory-mapped {n_samples} time series from {arrow_path} "
            f"(zero-copy, no RAM consumed at startup)"
        )

    # Train/val split: just index arrays, no data copied
    val_split = getattr(config, "val_split")
    np.random.seed(config.np_seed)
    n_val = int(n_samples * val_split)

    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Vectorized Arrow dataset: yields pre-formed, model-ready batches
    # directly from mmap'd buffers.
    config.data["batch_size"] = config.micro_batch_size  # DataLoader batch size is micro_batch_size
    train_dataset = Chronos2Dataset.from_arrow(
        table=table, indices=train_indices, mode="train", **config.data
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
    )

    val_dataloader = None
    if n_val > 0:
        val_dataset = Chronos2Dataset.from_arrow(
            table=table, indices=val_indices, mode="validation", **config.data
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            pin_memory=True,
            persistent_workers=True,
            num_workers=4,
            prefetch_factor=2,
        )

    if is_master:
        print("Batch size per GPU:", config.micro_batch_size * config.gradient_accumulation_steps)

    # Create trainer and start training
    trainer = Trainer(config, model, train_dataloader, val_dataloader=val_dataloader)
    model = trainer.train()


if __name__ == "__main__":
    set_sharing_strategy("file_system")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
