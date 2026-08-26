"""PatchTST-FM datasets

Data contract
-------------
Each batch yielded is a dict with the following keys, all torch.Tensor:

    inputs    : (B, T)
    pred_mask : (B, T)
    miss_mask : (B, T)
    pad_mask  : (B, T)

These map directly onto the keyword arguments of `PatchTSTFMForPretraining.forward`.

Masking strategy (Section 3.2 and Appendix A.3 of arXiv:2602.06909)
-------------------------------------------------------------------
For each training sample we build three sub-masks that are OR'd together as the
full `pred_mask` the model receives:

1. Contiguous Patch Masking (CPM): blocks of `N_cpm * d_patch` consecutive
   timestamps are masked at patch-aligned random positions.  The block count is
   chosen so that the masked ratio ≈ `mask_ratio`.  For short series the block
   size is reduced adaptively:
       N_cpm(x) = min(ceil(ceil(T(x)/d_patch) / 4), N_cpm)

2. Tail prediction mask: additionally, a random number of timesteps in
   `[0, 4 * N_cpm * d_patch)` are masked at the *end* of each sample to
   encourage forecasting.

3. Missing value mask: positions that were NaN in the raw input.

The loss is computed only on positions where `pred_mask=1 AND pad_mask=0 AND
miss_mask=0`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from tsfm.data.dataset import (
    BaseArrowFileDataset,
    BaseFileFormatGiftEvalDataset,
    BaseGiftEvalDataset,
    DatasetMode,
)


def _adaptive_cpm_block_size(n_valid_patches: int, n_cpm: int) -> int:
    """Adaptive CPM block-size (Appendix A.3).

    For a series with ``n_valid_patches`` non-padding patches the block size is
    reduced so that we never mask more than a quarter of the valid history.

        N_cpm(x) = min(ceil(n_valid_patches / 4), N_cpm)
    """
    return min(math.ceil(n_valid_patches / 4), n_cpm)


def _apply_cpm(
    length: int,
    d_patch: int,
    n_cpm: int,
    mask_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a CPM binary mask (1 = masked) of shape ``(length,)``.

    Works entirely at the patch level for efficiency:
      1. Compute how many non-overlapping blocks of size ``n_cpm`` are needed.
      2. Enumerate all valid block-start positions (multiples of ``n_cpm``),
         shuffle them, and take as many as needed — O(n_patch) with no loops or
         rejection sampling.
      3. Expand once: patch-level bool array → timestamp-level array via repeat.

    Parameters
    ----------
    length:
        Total sequence length (must be divisible by ``d_patch``).
    d_patch:
        Patch size in time-steps.
    n_cpm:
        Block size in *patches* (already adapted for the sequence length).
    mask_ratio:
        Target fraction of patches to mask.  The actual ratio will be the
        nearest multiple of ``n_cpm`` that does not exceed ``n_patch``.
    rng:
        NumPy random generator.
    """
    n_patch = length // d_patch

    # Number of non-overlapping blocks needed (rounded to nearest whole block)
    n_blocks_target = max(1, round(n_patch * mask_ratio / n_cpm))

    # All valid non-overlapping block-start positions (step = n_cpm ensures
    # blocks tile the sequence with no overlap by construction)
    block_starts = np.arange(0, n_patch - n_cpm + 1, n_cpm)  # shape (n_slots,)
    n_slots = len(block_starts)

    if n_slots == 0:
        # Degenerate: fewer patches than block size → mask everything
        return np.ones(length, dtype=bool)

    n_blocks = min(n_blocks_target, n_slots)

    # Choose which block slots to mask (no replacement → guaranteed non-overlap)
    chosen = rng.choice(n_slots, size=n_blocks, replace=False)
    chosen_starts = block_starts[chosen]  # patch indices

    # Build patch-level mask fully vectorised: for each chosen block start,
    # add the offsets [0, 1, ..., n_cpm-1].  outer_shape = (n_blocks, n_cpm),
    # then ravel and use as an index into a zeros array.
    offsets = np.arange(n_cpm)  # (n_cpm,)
    patch_indices = (chosen_starts[:, None] + offsets).ravel()  # (n_blocks * n_cpm,)
    patch_mask = np.zeros(n_patch, dtype=bool)
    patch_mask[patch_indices] = True

    # Expand patch-level mask → timestamp-level mask (single allocation)
    return np.repeat(patch_mask, d_patch)


def _apply_tail_mask(
    length: int,
    d_patch: int,
    n_cpm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly mask up to ``4 * n_cpm * d_patch`` timestamps at the tail.

    This extra tail mask pushes the model to learn forecasting (Appendix A.3).
    The tail length is sampled uniformly from ``[0, 4 * n_cpm * d_patch)``.
    """
    max_tail = 4 * n_cpm * d_patch
    tail_len = int(rng.integers(0, max_tail))
    tail_len = min(tail_len, length)
    mask = np.zeros(length, dtype=bool)
    if tail_len > 0:
        mask[-tail_len:] = True
    return mask


# ---------------------------------------------------------------------------
# Core dataset: Arrow-backed vectorised batch dataset
# ---------------------------------------------------------------------------


class PatchTSTFMArrowDataset(BaseArrowFileDataset):
    """Memory-mapped Arrow dataset that yields model-ready batches for PatchTST-FM.

    Parameters
    ----------
    arrow_path : str | Path
        Path to the Arrow IPC file containing the dataset.  Should have a "target" column.
    context_length : int
        Model context window ``T``.  All samples are left-padded or
        right-truncated to this length.
    d_patch : int
        Patch size.  ``context_length`` must be divisible by ``d_patch``.
    batch_size : int
        Number of time series per batch.
    mask_ratio : float
        Target fraction of *patches* to mask with CPM.  Default ``0.4``.
    n_cpm : int
        CPM block size in *patches*.  Default ``8``.
    min_past : int
        Minimum number of non-padding time-steps required for a sample to be
        included.  Samples shorter than this are dropped.  Default ``1``.
    mode : str | DatasetMode
        ``"train"`` (infinite random batches) or ``"validation"``
        (one-pass sequential batches).
    indices : np.ndarray | None
        Optional subset of row indices to use (e.g. for a train/val split).
    """

    def __init__(
        self,
        arrow_path: str | Path,
        context_length: int,
        d_patch: int,
        batch_size: int,
        mask_ratio: float = 0.4,
        n_cpm: int = 8,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        indices: np.ndarray | None = None,
    ) -> None:
        super().__init__(arrow_path, batch_size, mode, min_past, indices)

        assert context_length % d_patch == 0, (
            f"context_length ({context_length}) must be divisible by d_patch ({d_patch})."
        )

        self.context_length = context_length
        self.d_patch = d_patch
        self.n_patch = context_length // d_patch
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.n_cpm = n_cpm

    # ---------------------------------------------------------------------- #
    # Batch construction
    # ---------------------------------------------------------------------- #

    def _fetch_batch(self, task_indices: np.ndarray) -> dict[str, torch.Tensor]:
        """Build a complete model-ready batch (NumPy-dominant).

        Returns
        -------
        dict with keys ``inputs``, ``pred_mask``, ``miss_mask``, ``pad_mask``,
        all of shape ``(B, T)`` as ``torch.float32`` tensors.
        """
        bs = len(task_indices)
        T = self.context_length

        # Sort for sequential mmap page reads, then unsort
        order = np.argsort(task_indices)
        sorted_task_idx = task_indices[order]
        arrow_idx = self._valid_indices[sorted_task_idx]
        lengths = self._valid_lengths[sorted_task_idx]

        starts = self._offsets[arrow_idx]

        # Pre-allocate output buffers
        inputs_buf = np.zeros((bs, T), dtype=np.float32)
        pred_buf = np.zeros((bs, T), dtype=np.float32)
        miss_buf = np.zeros((bs, T), dtype=np.float32)
        pad_buf = np.zeros((bs, T), dtype=np.float32)

        rng = self._rng  # reuse instance RNG — no entropy-pool hit per batch

        for i in range(bs):
            L = int(lengths[i])
            s = int(starts[i])

            # ---------------------------------------------------------- #
            # 1. Read raw values and identify NaNs (missing mask)
            # ---------------------------------------------------------- #
            raw = self._values[s : s + L]
            if not self._values_f32:
                raw = raw.astype(np.float32)
            raw = np.ascontiguousarray(raw)  # ensure C-contiguous slice

            miss_i = np.isnan(raw)  # (L,)

            # ---------------------------------------------------------- #
            # 2. Random window selection (train) / last window (val)
            # ---------------------------------------------------------- #
            if self.mode == DatasetMode.TRAIN:
                max_end = L
                min_end = min(self.min_past, L)
                end = int(rng.integers(min_end, max_end + 1))
                ctx_start = max(0, end - T)
            else:
                end = L  # validation: use the full series
                ctx_start = max(0, end - T)

            ctx_len = end - ctx_start
            raw_ctx = raw[ctx_start:end]  # (ctx_len,)
            miss_ctx = miss_i[ctx_start:end]  # (ctx_len,)

            # ---------------------------------------------------------- #
            # 3. Left-pad to T
            # ---------------------------------------------------------- #
            pad_len = T - ctx_len
            # inputs: replace NaN with 0 (model zeros out masked positions anyway)
            x_in = np.concatenate([np.zeros(pad_len, dtype=np.float32), raw_ctx])
            x_in = np.where(np.isnan(x_in), np.float32(0.0), x_in)

            miss_full = np.concatenate([np.zeros(pad_len, dtype=bool), miss_ctx])
            pad_full = np.zeros(T, dtype=bool)
            if pad_len > 0:
                pad_full[:pad_len] = True

            # ---------------------------------------------------------- #
            # 4. CPM + tail prediction mask (applied only to non-padding)
            # ---------------------------------------------------------- #
            # Number of valid (non-padded) patches for adaptive block-size
            n_valid_patches = math.ceil(ctx_len / self.d_patch)
            eff_n_cpm = _adaptive_cpm_block_size(n_valid_patches, self.n_cpm)

            # Apply CPM on the full T window; pad positions won't be
            # selected because CPM uses patch-aligned starts and pad patches
            # will simply contribute 0 loss regardless.
            cpm_mask = _apply_cpm(T, self.d_patch, eff_n_cpm, self.mask_ratio, rng)

            tail_mask = _apply_tail_mask(T, self.d_patch, eff_n_cpm, rng)

            # Full prediction mask: CPM ∪ tail  (no masking of pure-pad positions)
            pred_full = (cpm_mask | tail_mask) & ~pad_full

            # ---------------------------------------------------------- #
            # 5. Fill batch buffers
            # ---------------------------------------------------------- #
            inputs_buf[i] = x_in
            pred_buf[i] = pred_full.astype(np.float32)
            miss_buf[i] = miss_full.astype(np.float32)
            pad_buf[i] = pad_full.astype(np.float32)

        # ------------------------------------------------------------------ #
        # Unsort back to original (random) order
        # ------------------------------------------------------------------ #
        inv_order = np.argsort(order)
        inputs_buf = inputs_buf[inv_order]
        pred_buf = pred_buf[inv_order]
        miss_buf = miss_buf[inv_order]
        pad_buf = pad_buf[inv_order]

        return {
            "inputs": torch.from_numpy(inputs_buf),
            "pred_mask": torch.from_numpy(pred_buf),
            "miss_mask": torch.from_numpy(miss_buf),
            "pad_mask": torch.from_numpy(pad_buf),
        }


class PatchTSTFMGiftEval(BaseFileFormatGiftEvalDataset):
    """Streaming dataset over a locally-downloaded GiftEvalPretrain directory.

    Expected layout on disk (mirrors the HuggingFace repo structure)

        <root>/
            electricity/
                H/
                    data-00000-of-00001.arrow
                D/
                    data-00000-of-00001.arrow
            m4_yearly/
                data-00000-of-00001.arrow
            solar/
                H/
                    data-00000-of-00001.arrow
                ...

    Each Arrow file is in IPC stream format (opened with
    `pyarrow.ipc.open_stream`).  Every row has a flat
    `target: list<float>` column — one row = one univariate time series.

    Paradigm
    `__init__` walks the directory tree, collects every `data-*.arrow`
    path, memory-maps each file, and stores the open `pyarrow.Table` handles
    in `self.datasets` — a plain Python list.  No heavy I/O at iteration time.

    `__iter__` samples rows from the list of tables.  In training mode a
    table is picked uniformly at random and a random row is drawn from it.
    In validation mode all rows of all tables are visited exactly once.
    Series are accumulated in a rolling buffer; a batch is yielded whenever
    the buffer reaches `batch_size`.

    Parameters
    ----------
    root : str | Path
        Path to the root directory of the downloaded GiftEvalPretrain data.
    context_length : int
        Model context window ``T``.
    d_patch : int
        Patch size.  ``context_length`` must be divisible by ``d_patch``.
    batch_size : int
        Number of univariate series per yielded batch.
    mask_ratio : float
        Target CPM masking ratio.  Default ``0.4``.
    n_cpm : int
        CPM block size in patches.  Default ``8``.
    min_past : int
        Minimum series length.  Shorter series are silently skipped.
        Default ``1``.
    dataset_names : list[str] | None
        Optional whitelist of top-level sub-dataset directory names
        (e.g. `["electricity", "m4_yearly"]`).  If None all sub-datasets
        found under `root` are used.
    arrow_paths: list[Path] | None
        Optional list of Arrow file paths to use.
    """

    def __init__(
        self,
        root: str | Path,
        context_length: int,
        d_patch: int,
        batch_size: int,
        mask_ratio: float = 0.4,
        n_cpm: int = 8,
        min_past: int = 1,
        dataset_names: list[str] | None = None,
        arrow_paths: list[Path] | None = None,
        exclude_datasets: list[str] | None = None,
        random_context_length: bool = True,
        drs_factors: list[int] | None = None,
        num_files_per_batch: int | None = None,
    ) -> None:
        super().__init__(root, batch_size, min_past, dataset_names, arrow_paths, exclude_datasets, num_files_per_batch)
        print(exclude_datasets)

        assert context_length % d_patch == 0, (
            f"context_length ({context_length}) must be divisible by d_patch ({d_patch})."
        )

        self.context_length = context_length
        self.d_patch = d_patch
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.n_cpm = n_cpm
        self.random_context_length = random_context_length

        # Diversity Resolution Sampling (DRS): sample one downsampling factor
        # per sample. Factors are configurable; probabilities are hardcoded.
        factors = np.array(drs_factors if drs_factors is not None else np.arange(1, 16), dtype=np.int64)
        if factors.ndim != 1 or len(factors) == 0:
            raise ValueError("drs_factors must be a non-empty 1D list of positive integers.")
        if np.any(factors < 1):
            raise ValueError("All drs_factors must be >= 1.")
        self.drs_factors = factors

        # Hardcoded probability law: p(r) ∝ 1/r (bias toward lighter downsampling).
        probs = 1.0 / self.drs_factors.astype(np.float64)
        self.drs_probabilities = probs / probs.sum()

    def _build_batch(
        self,
        sources: list[tuple[np.ndarray, np.ndarray, bool]],
        source_idx: list[int],
        flat_rows: list[int],
        lengths: list[int],
    ) -> dict[str, torch.Tensor]:
        """Build a model-ready batch from rows spread across multiple sources.

        Parameters
        ----------
        sources : list of (offsets, values, is_f32)
            One entry per open file.  All numpy views must still be alive
            (the caller keeps the pyarrow objects alive until this returns).
        source_idx : list[int]
            Which source each row comes from (parallel to flat_rows/lengths).
        flat_rows : list[int]
            Flat Arrow row index within the corresponding source's offsets.
        lengths : list[int]
            Series length for each entry.
        """
        T = self.context_length
        P = T // self.d_patch
        rng = self._rng

        bs = len(flat_rows)
        lengths_arr = np.array(lengths, dtype=np.int64)

        if self.random_context_length:
            # Sample a random context length per sample: U(64, T) to avoid
            # overfitting to a specific context length.
            sampled_ctx_lens = rng.integers(64, T + 1, size=bs)  # (bs,)
        else:
            sampled_ctx_lens = np.full(bs, T, dtype=np.int64)

        drs_arr = rng.choice(
            self.drs_factors,
            size=bs,
            replace=True,
            p=self.drs_probabilities,
        ).astype(np.int64)

        # Resampling should happen before padding. To keep an effective
        # (post-DRS) context length around sampled_ctx_lens, fetch a longer
        # pre-DRS window of sampled_ctx_lens * drs.
        pre_ctx_lens = np.minimum(lengths_arr, sampled_ctx_lens * drs_arr)

        # 1. Window-end selection (batch, one rng call)
        ends = lengths_arr.copy()
        long_mask = lengths_arr > pre_ctx_lens
        lo = pre_ctx_lens[long_mask]
        hi = lengths_arr[long_mask] + 1
        ends[long_mask] = rng.integers(lo, hi)

        # 2. Write directly into output buffers — no copy
        inputs_buf = np.zeros((bs, T), dtype=np.float32)
        miss_buf = np.zeros((bs, T), dtype=np.float32)
        pad_buf = np.zeros((bs, T), dtype=np.float32)
        ctx_lens = np.zeros(bs, dtype=np.int64)

        for i, fr in enumerate(flat_rows):
            offsets, values, is_f32 = sources[source_idx[i]]
            s = int(offsets[fr])  # Start
            e = int(ends[i])  # End
            cl = int(pre_ctx_lens[i])
            raw = values[s + e - cl : s + e]  # view into mmap, no copy
            if not is_f32:
                raw = raw.astype(np.float32)

            drs = int(drs_arr[i])
            if drs > 1:
                raw = raw[::drs]

            cl_eff = min(len(raw), T)
            ctx_lens[i] = cl_eff
            pl = T - cl_eff

            inputs_buf[i, pl:] = raw
            if pl:
                pad_buf[i, :pl] = 1.0

        # Batch NaN handling — one isnan call over the whole (bs, T) array
        nan_mask = np.isnan(inputs_buf)
        if nan_mask.any():
            inputs_buf[nan_mask] = 0.0
            miss_buf[nan_mask] = 1.0

        # 3. CPM masks
        # Per-sample adaptive block size, same formula as before but
        # applied element-wise instead of using the batch minimum.
        eff_n_cpm_arr = np.clip(
            np.ceil(ctx_lens / self.d_patch).astype(np.int64) // 4 + 1,
            1,
            self.n_cpm,
        )  # (bs,)

        # Group samples that share the same eff_n_cpm and run the original
        # vectorised CPM once per group — no 3-D arrays needed.
        # Iterate over the fixed range [1, n_cpm] instead of np.unique
        # (which sorts) since the values are already bounded integers.
        cpm_patch = np.zeros((bs, P), dtype=bool)
        for enc in range(1, self.n_cpm + 1):
            idx = np.where(eff_n_cpm_arr == enc)[0]
            if len(idx) == 0:
                continue
            g = len(idx)
            n_slots = max(1, P // enc)
            n_blocks = min(max(1, round(P * self.mask_ratio / enc)), n_slots)

            chosen_slots = np.argsort(rng.random((g, n_slots)), axis=1)[:, :n_blocks]
            chosen_patch = chosen_slots * enc  # (g, n_blocks)
            patch_idx = (chosen_patch[:, :, None] + np.arange(enc)).reshape(g, -1)
            assert patch_idx.max() < P, f"patch_idx OOB: max={patch_idx.max()}, P={P}, enc={enc}"
            cpm_patch[idx.reshape(-1, 1), patch_idx] = True

        cpm_mask = np.repeat(cpm_patch, self.d_patch, axis=1)

        # 4. Tail masks
        max_tail_arr = (4 * eff_n_cpm_arr * self.d_patch).astype(np.int64)
        tail_lens = rng.integers(0, max_tail_arr + 1)  # (bs,)
        tail_mask = np.arange(T)[None, :] >= (T - tail_lens[:, None])

        # 5. Combine
        pred_buf = ((cpm_mask | tail_mask) & (pad_buf == 0)).astype(np.float32)

        return {
            "inputs": torch.from_numpy(inputs_buf),
            "pred_mask": torch.from_numpy(pred_buf),
            "miss_mask": torch.from_numpy(miss_buf),
            "pad_mask": torch.from_numpy(pad_buf),
        }


class PatchTSTFMPatternRandomWalkDataset(IterableDataset):
    """On-the-fly synthetic dataset with repeated random-walk motifs.

    Each sample is generated by:
      1) sampling a motif length,
      2) drawing a random-walk motif,
      3) repeating it to fill the window,
            4) adding linear/quadratic/periodic trends and Gaussian noise,
            5) left-padding to the model context length.

    The yielded batch contract matches PatchTST pretraining:
    ``inputs``, ``pred_mask``, ``miss_mask``, ``pad_mask`` all with shape ``(B, T)``.
    """

    def __init__(
        self,
        context_length: int,
        d_patch: int,
        batch_size: int,
        mask_ratio: float = 0.4,
        n_cpm: int = 8,
        min_past: int = 1,
        random_context_length: bool = True,
        motif_len_min: int = 3,
        motif_len_max: int = 1024,
        trend_scale: float = 1.0,
        quadratic_trend_scale: float = 0.0,
        periodic_trend_scale: float = 0.0,
        periodic_period_min: int = 8,
        periodic_period_max: int = 256,
        noise_std_min: float = 0.02,
        noise_std_max: float = 0.2,
        amplitude_min: float = 0.5,
        amplitude_max: float = 2.0,
    ) -> None:
        super().__init__()

        assert context_length % d_patch == 0, (
            f"context_length ({context_length}) must be divisible by d_patch ({d_patch})."
        )
        if motif_len_min < 2 or motif_len_max < motif_len_min:
            raise ValueError("motif_len_min/motif_len_max must satisfy 2 <= min <= max.")
        if noise_std_min < 0 or noise_std_max < noise_std_min:
            raise ValueError("noise_std_min/noise_std_max must satisfy 0 <= min <= max.")
        if amplitude_min <= 0 or amplitude_max < amplitude_min:
            raise ValueError("amplitude_min/amplitude_max must satisfy 0 < min <= max.")
        if periodic_period_min < 2 or periodic_period_max < periodic_period_min:
            raise ValueError(
                "periodic_period_min/periodic_period_max must satisfy 2 <= min <= max."
            )
        if quadratic_trend_scale < 0:
            raise ValueError("quadratic_trend_scale must be >= 0.")
        if periodic_trend_scale < 0:
            raise ValueError("periodic_trend_scale must be >= 0.")

        self.context_length = context_length
        self.d_patch = d_patch
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.n_cpm = n_cpm
        self.min_past = min_past
        self.random_context_length = random_context_length

        self.motif_len_min = motif_len_min
        self.motif_len_max = motif_len_max
        self.trend_scale = trend_scale
        self.quadratic_trend_scale = quadratic_trend_scale
        self.periodic_trend_scale = periodic_trend_scale
        self.periodic_period_min = periodic_period_min
        self.periodic_period_max = periodic_period_max
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max

        self._rng: np.random.Generator | None = None
    

    def __len__(self):
        return 10_000_000

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        self._rng = np.random.default_rng(seed)
        while True:
            yield self._build_batch()

    def _generate_pattern_random_walk(self, length: int) -> np.ndarray:
        rng = self._rng
        motif_len = int(rng.integers(self.motif_len_min, self.motif_len_max + 1))
        motif_len = min(motif_len, length)

        motif = np.cumsum(rng.standard_normal(motif_len)).astype(np.float32)

        repeats = int(np.ceil(length / motif_len))
        base = np.tile(motif, repeats)[:length].astype(np.float32)

        amplitude = float(rng.uniform(self.amplitude_min, self.amplitude_max))
        t = np.linspace(0.0, 1.0, num=length, dtype=np.float32)

        # Linear trend
        lin_coef = float(rng.uniform(-self.trend_scale, self.trend_scale))
        lin_trend = lin_coef * t

        # Quadratic trend (mean-centered to avoid systematic offsets)
        quad_basis = t**2 - np.mean(t**2)
        quad_coef = float(rng.uniform(-self.quadratic_trend_scale, self.quadratic_trend_scale))
        quad_trend = quad_coef * quad_basis

        # Periodic trend
        period = int(rng.integers(self.periodic_period_min, self.periodic_period_max + 1))
        period = min(period, max(2, length))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        periodic_amp = float(rng.uniform(0.0, self.periodic_trend_scale))
        periodic_trend = periodic_amp * np.sin(
            2.0 * np.pi * np.arange(length, dtype=np.float32) / period + phase
        )

        trend = lin_trend + quad_trend + periodic_trend

        noise_std = float(rng.uniform(self.noise_std_min, self.noise_std_max))
        noise = rng.normal(0.0, noise_std, size=length).astype(np.float32)

        return amplitude * base + trend + noise

    def _build_batch(self) -> dict[str, torch.Tensor]:
        T = self.context_length
        P = T // self.d_patch
        rng = self._rng
        bs = self.batch_size

        if self.random_context_length:
            target_ctx_lens = rng.integers(max(1, min(64, T)), T + 1, size=bs)
        else:
            target_ctx_lens = np.full(bs, T, dtype=np.int64)

        raw_ctx_lens = np.maximum(self.min_past, target_ctx_lens).astype(np.int64)

        inputs_buf = np.zeros((bs, T), dtype=np.float32)
        miss_buf = np.zeros((bs, T), dtype=np.float32)
        pad_buf = np.zeros((bs, T), dtype=np.float32)
        ctx_lens = np.zeros(bs, dtype=np.int64)

        for i in range(bs):
            raw = self._generate_pattern_random_walk(int(raw_ctx_lens[i]))

            cl_eff = min(len(raw), T)
            ctx_lens[i] = cl_eff
            pl = T - cl_eff

            inputs_buf[i, pl:] = raw[-cl_eff:]
            if pl:
                pad_buf[i, :pl] = 1.0

        eff_n_cpm_arr = np.clip(
            np.ceil(ctx_lens / self.d_patch).astype(np.int64) // 4 + 1,
            1,
            self.n_cpm,
        )

        cpm_patch = np.zeros((bs, P), dtype=bool)
        for enc in range(1, self.n_cpm + 1):
            idx = np.where(eff_n_cpm_arr == enc)[0]
            if len(idx) == 0:
                continue
            g = len(idx)
            n_slots = max(1, P // enc)
            n_blocks = min(max(1, round(P * self.mask_ratio / enc)), n_slots)

            chosen_slots = np.argsort(rng.random((g, n_slots)), axis=1)[:, :n_blocks]
            chosen_patch = chosen_slots * enc
            patch_idx = (chosen_patch[:, :, None] + np.arange(enc)).reshape(g, -1)
            cpm_patch[idx.reshape(-1, 1), patch_idx] = True

        cpm_mask = np.repeat(cpm_patch, self.d_patch, axis=1)
        max_tail_arr = (4 * eff_n_cpm_arr * self.d_patch).astype(np.int64)
        tail_lens = rng.integers(0, max_tail_arr + 1)
        tail_mask = np.arange(T)[None, :] >= (T - tail_lens[:, None])
        pred_buf = ((cpm_mask | tail_mask) & (pad_buf == 0)).astype(np.float32)

        return {
            "inputs": torch.from_numpy(inputs_buf),
            "pred_mask": torch.from_numpy(pred_buf),
            "miss_mask": torch.from_numpy(miss_buf),
            "pad_mask": torch.from_numpy(pad_buf),
        }
