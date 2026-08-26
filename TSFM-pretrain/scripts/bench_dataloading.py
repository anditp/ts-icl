from __future__ import annotations

import argparse
import copy
import csv
import gc
import itertools
import os
import pathlib
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.multiprocessing import set_sharing_strategy, set_start_method
from torch.utils.data import DataLoader

import tsfm.train.trainer as _trainer_module
from tsfm.model.chronos2.chronos2 import Chronos2Model
from tsfm.model.chronos2.config import Chronos2CoreConfig
from tsfm.model.chronos2.dataset import Chronos2Dataset
from tsfm.train.trainer import Trainer


def generate_dummy_data(
    n_series: int = 500,
    min_len: int = 100,
    max_len: int = 500,
    seed: int = 42,
) -> list[dict[str, np.ndarray]]:
    """Generate random univariate time series as list-of-dicts."""
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_series):
        length = rng.integers(min_len, max_len + 1)
        target = np.cumsum(rng.standard_normal(length)).astype(np.float32)
        data.append({"target": target})
    return data


def load_config(path: str | pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_one(
    cfg_dict: dict,
    batch_size: int,
    num_workers: int,
    dummy_data: list[dict[str, np.ndarray]],
    context_length: int,
    total_points: int,
) -> dict:
    """Run a single benchmark for one (batch_size, num_workers) pair.

    Creates a fresh model, dataset, dataloader and Trainer, trains for the
    computed number of steps, then returns timing results.

    Gradient accumulation is disabled: ``micro_batch_size == per_gpu_bs``,
    so exactly 1 dataloader fetch happens per optimizer step.

    In DDP mode (launched via ``torchrun``), the Trainer divides
    ``config.batch_size`` by ``world_size`` internally.  We set
    ``data.batch_size`` (what each DataLoader yields) to the per-GPU
    batch size so that each step processes the correct amount of data
    across all ranks.
    """
    cfg = copy.deepcopy(cfg_dict)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    max_steps = total_points // (batch_size * context_length)
    if max_steps < 1:
        raise ValueError(
            f"total_points={total_points} too small for "
            f"batch_size={batch_size}, context_length={context_length}"
        )

    # Batch_size should be the total across all GPUs.
    # BS = MBS * world_size * grad_accum_steps, but we disable grad_accum so MBS = BS / world_size.
    # data[BS] should always be MBS
    per_gpu_bs = cfg["micro_batch_size"]

    cfg["batch_size"] = per_gpu_bs * world_size  # total batch size across all GPUs
    cfg["max_steps"] = max_steps
    cfg["data"]["batch_size"] = per_gpu_bs

    cfg["wandb_log"] = False
    cfg["mlflow_log"] = False
    cfg["resume_training"] = False
    cfg["save_temp_every"] = max_steps + 1
    cfg["save_perm_every"] = max_steps + 1
    cfg["val_every"] = max_steps + 1  # skip validation
    cfg["val_split"] = 0.0
    cfg["print_every"] = max(1, max_steps // 4)

    ckpt_dir = tempfile.mkdtemp(prefix="bench_ckpt_")
    cfg["checkpoint_dir"] = ckpt_dir

    config = SimpleNamespace(**cfg)

    core_config = Chronos2CoreConfig(**config.model)
    model = Chronos2Model(core_config)

    dataset = Chronos2Dataset(inputs=dummy_data, mode="train", **config.data)
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # dataset yields pre-formed batches
        num_workers=num_workers,
        pin_memory="cuda" in config.device,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # --- Warm up DataLoader workers -------------------------------------
    # With persistent_workers=True the worker processes survive iterator
    # resets, so spawning them once here means Trainer.train() reuses the
    # same pool.  This keeps spawn overhead out of the timing.
    if num_workers > 0:
        warmup_it = iter(dataloader)
        next(warmup_it)  # forces worker spawn
        del warmup_it

    torch.manual_seed(config.torch_seed)
    np.random.seed(config.np_seed)

    t0_total = time.perf_counter()
    trainer = Trainer(config, model, dataloader)
    t0_train = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    elapsed_total = t1 - t0_total  # init + training
    elapsed_train = t1 - t0_train  # training only

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    del trainer, model, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_gpus": num_gpus,
        "world_size": world_size,
        "per_gpu_bs": per_gpu_bs,
        "max_steps": max_steps,
        "total_points": total_points,
        "total_s": round(elapsed_total, 2),
        "train_s": round(elapsed_train, 2),
        "pts/s_total": round(total_points / elapsed_total, 0),
        "pts/s_train": round(total_points / elapsed_train, 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput across (batch_size, num_workers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config (default: configs/test.yaml)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num-workers",
        nargs="+",
        type=int,
        default=[0, 2, 4],
        help="DataLoader num_workers values to benchmark",
    )
    parser.add_argument(
        "--total-points",
        type=int,
        default=None,
        help=(
            "Fixed total data points. Default: batch_size x max_steps x context_length from config"
        ),
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=500,
        help="Number of random time series to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results as CSV (e.g. results.csv)",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_master = rank == 0

    cfg_dict = load_config(args.config)

    if args.device is not None:
        cfg_dict["device"] = args.device

    context_length = cfg_dict["model"]["chronos_config"]["context_length"]

    # Default total_points: reference config's batch_size x max_steps x context_length
    if args.total_points is None:
        ref_bs = cfg_dict["batch_size"]
        ref_steps = cfg_dict["max_steps"]
        total_points = ref_bs * ref_steps * context_length
    else:
        total_points = args.total_points

    if is_master:
        print("=" * 70)
        print("  Dataloading / Training Throughput Benchmark")
        print("=" * 70)
        print(f"  Config:           {args.config}")
        print(f"  Device:           {cfg_dict['device']}")
        print(f"  Num GPUs:         {torch.cuda.device_count()}")
        print(f"  World size:       {world_size}")
        print(f"  Context length:   {context_length}")
        print(f"  Total points:     {total_points:,}")
        print(f"  Batch sizes:      {args.batch_sizes}")
        print(f"  Num workers:      {args.num_workers}")
        print(f"  N series:         {args.n_series}")
        print("=" * 70)
        print()

    set_sharing_strategy("file_system")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    dummy_data = generate_dummy_data(n_series=args.n_series)

    # -- Stable DDP: init once, prevent Trainer from re-init/destroy -------
    # The Trainer's @ddp_cleanup decorator calls destroy_process_group()
    # after each train(), and configure_ddp() calls init_process_group()
    # on construction.  Re-initializing NCCL repeatedly is fragile, so we:
    #   1) init the process group once here
    #   2) monkey-patch the trainer module's references to no-ops
    #   3) restore & destroy once after the benchmark loop
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        _real_init_pg = _trainer_module.init_process_group
        _real_destroy_pg = _trainer_module.destroy_process_group
        _trainer_module.init_process_group = lambda *a, **kw: None
        _trainer_module.destroy_process_group = lambda *a, **kw: None

    grid = list(itertools.product(args.batch_sizes, args.num_workers))
    results: list[dict] = []

    for i, (bs, nw) in enumerate(grid, 1):
        max_steps = total_points // (bs * context_length)
        if is_master:
            print(f"[{i}/{len(grid)}] batch_size={bs:>5}, num_workers={nw}, max_steps={max_steps}")
        result = run_one(
            cfg_dict,
            bs,
            nw,
            dummy_data,
            context_length,
            total_points,
        )
        results.append(result)
        if is_master:
            print(
                f"         -> total {result['total_s']:.2f}s ({result['pts/s_total']:,.0f} pts/s)  "
                f"| train {result['train_s']:.2f}s ({result['pts/s_train']:,.0f} pts/s)\n"
            )

    if ddp:
        _trainer_module.init_process_group = _real_init_pg
        _trainer_module.destroy_process_group = _real_destroy_pg
        dist.destroy_process_group()

    if is_master:
        header = (
            f"{'batch_size':>12} {'num_workers':>12} {'num_gpus':>10} {'world_size':>12} "
            f"{'per_gpu_bs':>12} {'max_steps':>12} "
            f"{'total_s':>10} {'pts/s_total':>14} "
            f"{'train_s':>10} {'pts/s_train':>14}"
        )
        sep = "-" * len(header)
        print("\n" + "=" * len(header))
        print("  RESULTS")
        print("=" * len(header))
        print(header)
        print(sep)
        for r in results:
            print(
                f"{r['batch_size']:>12} "
                f"{r['num_workers']:>12} "
                f"{r['num_gpus']:>10} "
                f"{r['world_size']:>12} "
                f"{r['per_gpu_bs']:>12} "
                f"{r['max_steps']:>12} "
                f"{r['total_s']:>10.2f} "
                f"{r['pts/s_total']:>14,.0f} "
                f"{r['train_s']:>10.2f} "
                f"{r['pts/s_train']:>14,.0f}"
            )
        print("=" * len(header))

        best = max(results, key=lambda r: r["pts/s_train"])
        print(
            f"\n  Best (train): batch_size={best['batch_size']}, "
            f"num_workers={best['num_workers']} "
            f"-> {best['pts/s_train']:,.0f} pts/s"
        )
        best_total = max(results, key=lambda r: r["pts/s_total"])
        print(
            f"  Best (total): batch_size={best_total['batch_size']}, "
            f"num_workers={best_total['num_workers']} "
            f"-> {best_total['pts/s_total']:,.0f} pts/s"
        )

        if args.output_csv:
            csv_path = pathlib.Path(args.output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
