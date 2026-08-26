#!/usr/bin/env python3
"""Evaluate periodic pattern synthetic datasets against checkpoints.

This script mirrors the behavior used in the other time_series_token project
evaluate_pattern utility but adapted for this repository. It can build a
dataset bundle of periodic motifs (from GiftEvalPretrain or a periodic
random-walk generator), evaluate PatchTST-FM pretraining checkpoints on the
synthetic datasets, and produce example plots.

Usage examples:
  - Build dataset bundle only:
      python scripts/evaluate_pattern.py --config configs/patchtst_fm.yaml \
          --dataset_file data/pattern_bundle.pt --build_dataset_only

  - Evaluate checkpoints in a directory:
      python scripts/evaluate_pattern.py --config configs/patchtst_fm.yaml \
          --dataset_file data/pattern_bundle.pt --checkpoint_dir checkpoints/calm-mullet \
          --output_csv results/pattern_eval.csv --plot_predictions_dir results/pred_plots

"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from tsfm.model.patchtst_fm.config import PatchTSTFMConfig
from tsfm.model.patchtst_fm.patchtst_fm import PatchTSTFMForPretraining


def strip_ddp_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    return out


def random_walk_batch(
    n_samples: int,
    context_length: int,
    valid_length: int,
    mask_ratio: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Original periodic random-walk generator (kept for compatibility)."""
    g = torch.Generator().manual_seed(seed)

    period = int(valid_length)
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    steps = torch.randn(n_samples, period, generator=g)
    motif = torch.cumsum(steps, dim=1)  # (N, period)
    reps = (context_length + period - 1) // period
    inputs = motif.repeat(1, reps)[:, :context_length].contiguous()

    pad_mask = torch.zeros(n_samples, context_length, dtype=torch.bool)
    miss_mask = torch.zeros(n_samples, context_length, dtype=torch.bool)
    pred_mask = torch.rand(n_samples, context_length, generator=g) < mask_ratio

    for i in range(n_samples):
        if not pred_mask[i].any():
            j = torch.randint(0, context_length, (1,), generator=g).item()
            pred_mask[i, j] = True

    return {
        "inputs": inputs,
        "pred_mask": pred_mask,
        "miss_mask": miss_mask,
        "pad_mask": pad_mask,
    }


def _collect_arrow_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise RuntimeError(f"GiftEval root does not exist: {root_path}")
    files = sorted(root_path.rglob("data-*.arrow"))
    if not files:
        raise RuntimeError(f"No data-*.arrow files found under: {root_path}")
    return files


def _sample_pattern_from_gifteval(
    arrow_files: list[Path],
    period: int,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> np.ndarray:
    """Sample one real motif of length `period` from GiftEvalPretrain files."""
    for _ in range(max_tries):
        path = arrow_files[int(rng.integers(0, len(arrow_files)))]

        mmap = pa.memory_map(str(path), "r")
        reader = pa_ipc.open_file(mmap)
        n_batches = reader.num_record_batches
        if n_batches == 0:
            mmap.close()
            continue

        b = int(rng.integers(0, n_batches))
        rb = reader.get_batch(b)
        col = rb.column("target")

        if pa.types.is_fixed_size_list(col.type):
            c = col.type.list_size
            inner = col.values
            offsets = inner.offsets.to_numpy()
            values = inner.values.to_numpy()
            lengths = np.diff(offsets[::c]).astype(np.int64)
            valid_rows = np.where(lengths >= period)[0]
            if len(valid_rows) == 0:
                mmap.close()
                continue
            r = int(valid_rows[int(rng.integers(0, len(valid_rows)))])
            ch = int(rng.integers(0, c))
            s = int(offsets[r * c + ch])
            e = int(offsets[r * c + ch + 1])
            series = values[s:e]
        else:
            offsets = col.offsets.to_numpy()
            values = col.values.to_numpy()
            lengths = np.diff(offsets).astype(np.int64)
            valid_rows = np.where(lengths >= period)[0]
            if len(valid_rows) == 0:
                mmap.close()
                continue
            r = int(valid_rows[int(rng.integers(0, len(valid_rows)))])
            s = int(offsets[r])
            e = int(offsets[r + 1])
            series = values[s:e]

        mmap.close()

        L = len(series)
        if L < period:
            continue
        start = int(rng.integers(0, L - period + 1))
        motif = np.asarray(series[start : start + period], dtype=np.float32)
        motif = np.nan_to_num(motif, nan=0.0, posinf=0.0, neginf=0.0)
        return motif

    raise RuntimeError(
        f"Could not sample a motif of length {period} from GiftEvalPretrain after {max_tries} tries"
    )


def gifteval_periodic_pattern_batch(
    n_samples: int,
    context_length: int,
    valid_length: int,
    mask_ratio: float,
    arrow_files: list[Path],
    seed: int,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    # valid_length is used as the pattern period (d_patch or 2*d_patch).
    period = int(valid_length)
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    # Build one real GiftEval motif per sample, then repeat periodically.
    motifs = np.stack(
        [_sample_pattern_from_gifteval(arrow_files, period, rng) for _ in range(n_samples)],
        axis=0,
    )
    motif = torch.from_numpy(motifs)
    reps = (context_length + period - 1) // period
    inputs = motif.repeat(1, reps)[:, :context_length].contiguous()

    # No padding/missing in the synthetic benchmark; full context is valid.
    pad_mask = torch.zeros(n_samples, context_length, dtype=torch.bool)
    miss_mask = torch.zeros(n_samples, context_length, dtype=torch.bool)
    pred_mask = torch.rand(n_samples, context_length, generator=g) < mask_ratio

    for i in range(n_samples):
        if not pred_mask[i].any():
            j = torch.randint(0, context_length, (1,), generator=g).item()
            pred_mask[i, j] = True

    return {
        "inputs": inputs,
        "pred_mask": pred_mask,
        "miss_mask": miss_mask,
        "pad_mask": pad_mask,
    }


def build_and_save_dataset_bundle(
    output_file: str,
    pattern_source: str,
    gifteval_root: str,
    context_length: int,
    d_patch: int,
    num_samples: int,
    mask_ratio: float,
    seed: int,
) -> None:
    source = pattern_source.lower()
    if source not in {"gifteval", "random_walk"}:
        raise ValueError(f"Unsupported pattern_source: {pattern_source}")

    if source == "gifteval":
        arrow_files = _collect_arrow_files(gifteval_root)
        ds_patch = gifteval_periodic_pattern_batch(
            n_samples=num_samples,
            context_length=context_length,
            valid_length=d_patch,
            mask_ratio=mask_ratio,
            arrow_files=arrow_files,
            seed=seed,
        )
        ds_2patch = gifteval_periodic_pattern_batch(
            n_samples=num_samples,
            context_length=context_length,
            valid_length=2 * d_patch,
            mask_ratio=mask_ratio,
            arrow_files=arrow_files,
            seed=seed + 1,
        )
        source_desc = "GiftEvalPretrain motifs"
    else:
        ds_patch = random_walk_batch(
            n_samples=num_samples,
            context_length=context_length,
            valid_length=d_patch,
            mask_ratio=mask_ratio,
            seed=seed,
        )
        ds_2patch = random_walk_batch(
            n_samples=num_samples,
            context_length=context_length,
            valid_length=2 * d_patch,
            mask_ratio=mask_ratio,
            seed=seed + 1,
        )
        source_desc = "Periodic random-walk motifs"

    bundle = {
        "meta": {
            "source": source_desc,
            "pattern_source": source,
            "gifteval_root": str(gifteval_root),
            "context_length": context_length,
            "d_patch": d_patch,
            "num_samples": num_samples,
            "mask_ratio": mask_ratio,
            "seed": seed,
        },
        "pat_len_patch": ds_patch,
        "pat_len_2patch": ds_2patch,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save(bundle, output_file)
    print(f"Saved dataset bundle: {output_file}")


def resolve_bundle_datasets(bundle: Dict) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (patch, 2patch) datasets from either new or legacy bundle keys."""
    if "pat_len_patch" in bundle and "pat_len_2patch" in bundle:
        return bundle["pat_len_patch"], bundle["pat_len_2patch"]
    if "rw_len_patch" in bundle and "rw_len_2patch" in bundle:
        return bundle["rw_len_patch"], bundle["rw_len_2patch"]
    raise KeyError(
        "Dataset bundle missing expected keys. Expected either "
        "('pat_len_patch','pat_len_2patch') or ('rw_len_patch','rw_len_2patch')."
    )


def plot_dataset_examples(bundle: Dict, output_file: str) -> None:
    """Plot one sample from each periodic dataset."""
    meta = bundle["meta"]
    d_patch = int(meta["d_patch"])
    source_desc = str(meta.get("source", "patterns"))

    ds_patch, ds_2patch = resolve_bundle_datasets(bundle)
    x1 = ds_patch["inputs"][0].cpu().numpy()
    x2 = ds_2patch["inputs"][0].cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x1, lw=1.5)
    axes[0].set_title(f"{source_desc} (period = d_patch = {d_patch})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x2, lw=1.5, color="tab:orange")
    axes[1].set_title(f"{source_desc} (period = 2*d_patch = {2 * d_patch})")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("time index")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved example plot: {output_file}")


@torch.no_grad()
def get_single_sample_prediction(
    model: PatchTSTFMForPretraining,
    dataset: Dict[str, torch.Tensor],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    sample_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (target, median_prediction, pred_mask) for one sample."""
    x = dataset["inputs"][sample_idx : sample_idx + 1].to(device)
    pm = dataset["pred_mask"][sample_idx : sample_idx + 1].to(device)
    mm = dataset["miss_mask"][sample_idx : sample_idx + 1].to(device)
    pad = dataset["pad_mask"][sample_idx : sample_idx + 1].to(device)

    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=use_amp and device.startswith("cuda"),
    ):
        out = model(inputs=x, pred_mask=pm, miss_mask=mm, pad_mask=pad)

    q = out["quantile_predictions"]
    # Expected shape in this codebase: (B, Q, T)
    median_idx = q.shape[1] // 2
    pred = q[0, median_idx].detach().cpu().numpy()
    target = x[0].detach().cpu().numpy()
    pred_mask = pm[0].detach().cpu().numpy().astype(bool)
    return target, pred, pred_mask


def plot_prediction_examples(
    ckpt_step: int,
    d_patch: int,
    patch_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    patch2_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_file: str,
    last_k: int = 0,
) -> None:
    """Plot model median prediction against target for both synthetic sets."""
    x1, y1, m1 = patch_triplet
    x2, y2, m2 = patch2_triplet

    # Optionally plot only the last K timesteps
    if last_k and last_k > 0:
        x1 = x1[-last_k:]
        y1 = y1[-last_k:]
        m1 = m1[-last_k:]
        x2 = x2[-last_k:]
        y2 = y2[-last_k:]
        m2 = m2[-last_k:]

    t1 = np.arange(len(x1))
    t2 = np.arange(len(x2))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t1, x1, lw=1.5, label="target")
    axes[0].plot(t1, y1, lw=1.25, label="pred (median)")
    axes[0].scatter(t1[m1], x1[m1], s=8, alpha=0.55, label="pred_mask")
    axes[0].set_title(f"step={ckpt_step} | period=d_patch={d_patch}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(t2, x2, lw=1.5, label="target")
    axes[1].plot(t2, y2, lw=1.25, label="pred (median)")
    axes[1].scatter(t2[m2], x2[m2], s=8, alpha=0.55, label="pred_mask")
    axes[1].set_title(f"step={ckpt_step} | period=2*d_patch={2 * d_patch}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("time index")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved prediction plot: {output_file}")


@torch.no_grad()
def evaluate_dataset(
    model: PatchTSTFMForPretraining,
    dataset: Dict[str, torch.Tensor],
    batch_size: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float]:
    n = dataset["inputs"].shape[0]
    losses = []
    maes = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = {k: v[s:e].to(device, non_blocking=True) for k, v in dataset.items()}

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp and device.startswith("cuda"),
        ):
            out = model(**batch)

        losses.append(float(out["loss"].detach().cpu()))
        maes.append(float(out["mae_loss"].detach().cpu()))

    return sum(losses) / len(losses), sum(maes) / len(maes)


def extract_step(path: str) -> int:
    m = re.search(r"step-(\d+)\.ckpt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_model_state_from_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in ckpt:
        sd = ckpt["model_state"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    return strip_ddp_prefixes(sd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to a training checkpoint (.ckpt) or HF directory to evaluate",
    )
    p.add_argument(
        "--pattern_source",
        type=str,
        choices=["random_walk", "gifteval"],
        default="gifteval",
        help="Data source for periodic motifs: original random_walk or gifteval snippets.",
    )
    p.add_argument(
        "--patch-lengths",
        type=int,
        nargs=2,
        default=None,
        help="Two pattern lengths to evaluate, e.g. --patch-lengths 16 32. Defaults to model d_patch and 2*d_patch.",
    )
    p.add_argument(
        "--gifteval_root",
        type=str,
        default="/data/parietal/store4/data/GiftEvalPretrain_nostream",
    )
    p.add_argument("--build_dataset_only", action="store_true")
    p.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help="Optional path to save the generated dataset bundle",
    )
    p.add_argument("--output_csv", type=str, default="checkpoints_synth_eval.csv")
    p.add_argument("--plot_examples_file", type=str, default=None)
    p.add_argument(
        "--plot_predictions_dir",
        type=str,
        default=None,
        help="If set, save per-checkpoint prediction plots in this directory.",
    )
    p.add_argument(
        "--plot_prediction_sample_idx",
        type=int,
        default=0,
        help="Sample index in precomputed dataset used for prediction plots.",
    )
    p.add_argument(
        "--plot-last-k",
        type=int,
        default=0,
        help="If >0, when saving prediction plots only plot the last K timesteps (e.g. 1000)",
    )

    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = PatchTSTFMConfig(**cfg["model"])
    context_length = int(model_cfg.context_length)
    config_d_patch = int(model_cfg.d_patch)

    # Determine pattern lengths to evaluate
    if args.patch_lengths is not None:
        pat_len, pat2_len = int(args.patch_lengths[0]), int(args.patch_lengths[1])
    else:
        pat_len = config_d_patch
        pat2_len = 2 * config_d_patch

    # Build datasets in-memory (no extra files required) unless the user
    # explicitly requested saving the bundle.
    if args.pattern_source == "gifteval":
        arrow_files = _collect_arrow_files(args.gifteval_root)
        ds_patch = gifteval_periodic_pattern_batch(
            n_samples=args.num_samples,
            context_length=context_length,
            valid_length=pat_len,
            mask_ratio=args.mask_ratio,
            arrow_files=arrow_files,
            seed=args.seed,
        )
        ds_2patch = gifteval_periodic_pattern_batch(
            n_samples=args.num_samples,
            context_length=context_length,
            valid_length=pat2_len,
            mask_ratio=args.mask_ratio,
            arrow_files=arrow_files,
            seed=args.seed + 1,
        )
    else:
        ds_patch = random_walk_batch(
            n_samples=args.num_samples,
            context_length=context_length,
            valid_length=pat_len,
            mask_ratio=args.mask_ratio,
            seed=args.seed,
        )
        ds_2patch = random_walk_batch(
            n_samples=args.num_samples,
            context_length=context_length,
            valid_length=pat2_len,
            mask_ratio=args.mask_ratio,
            seed=args.seed + 1,
        )

    if args.plot_examples_file is not None:
        bundle = {
            "meta": {
                "source": "GiftEvalPretrain motifs" if args.pattern_source == "gifteval" else "Periodic random-walk motifs",
                "pattern_source": args.pattern_source,
                "gifteval_root": args.gifteval_root,
                "context_length": context_length,
                "d_patch": pat_len,
                "d_patch2": pat2_len,
                "num_samples": args.num_samples,
                "mask_ratio": args.mask_ratio,
                "seed": args.seed,
            },
            "pat_len_patch": ds_patch,
            "pat_len_2patch": ds_2patch,
        }
        plot_dataset_examples(bundle, args.plot_examples_file)

    if args.save_dataset:
        bundle = {
            "meta": {
                "source": "GiftEvalPretrain motifs" if args.pattern_source == "gifteval" else "Periodic random-walk motifs",
                "pattern_source": args.pattern_source,
                "gifteval_root": args.gifteval_root,
                "context_length": context_length,
                "d_patch": pat_len,
                "d_patch2": pat2_len,
                "num_samples": args.num_samples,
                "mask_ratio": args.mask_ratio,
                "seed": args.seed,
            },
            "pat_len_patch": ds_patch,
            "pat_len_2patch": ds_2patch,
        }
        os.makedirs(os.path.dirname(args.save_dataset) or ".", exist_ok=True)
        torch.save(bundle, args.save_dataset)
        print(f"Saved dataset bundle: {args.save_dataset}")

    # Load and evaluate single checkpoint specified by --ckpt-path
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    model = PatchTSTFMForPretraining(model_cfg).to(args.device)
    model.eval()

    amp = bool(cfg.get("amp", True))
    dtype_name = str(cfg.get("dtype", "float16")).lower()
    amp_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

    step = extract_step(args.ckpt_path)
    sd = load_model_state_from_ckpt(args.ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"[step {step}] missing keys: {len(missing)}")
    if unexpected:
        print(f"[step {step}] unexpected keys: {len(unexpected)}")

    l1, m1 = evaluate_dataset(
        model=model,
        dataset=ds_patch,
        batch_size=args.eval_batch_size,
        device=args.device,
        use_amp=amp,
        amp_dtype=amp_dtype,
    )
    l2, m2 = evaluate_dataset(
        model=model,
        dataset=ds_2patch,
        batch_size=args.eval_batch_size,
        device=args.device,
        use_amp=amp,
        amp_dtype=amp_dtype,
    )

    if args.plot_predictions_dir is not None:
        triplet_patch = get_single_sample_prediction(
            model=model,
            dataset=ds_patch,
            device=args.device,
            use_amp=amp,
            amp_dtype=amp_dtype,
            sample_idx=args.plot_prediction_sample_idx,
        )
        triplet_2patch = get_single_sample_prediction(
            model=model,
            dataset=ds_2patch,
            device=args.device,
            use_amp=amp,
            amp_dtype=amp_dtype,
            sample_idx=args.plot_prediction_sample_idx,
        )
        pred_plot_path = os.path.join(args.plot_predictions_dir, f"step-{step:06d}.png")
        plot_prediction_examples(
            ckpt_step=step,
            d_patch=pat_len,
            patch_triplet=triplet_patch,
            patch2_triplet=triplet_2patch,
            output_file=pred_plot_path,
            last_k=args.plot_last_k,
        )

    row = {
        "step": step,
        "checkpoint": args.ckpt_path,
        f"loss_pat_len_{pat_len}": l1,
        f"mae_pat_len_{pat_len}": m1,
        f"loss_pat_len_{pat2_len}": l2,
        f"mae_pat_len_{pat2_len}": m2,
    }

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(row)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
