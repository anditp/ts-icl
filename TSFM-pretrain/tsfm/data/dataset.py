from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from torch.utils.data import IterableDataset


class DatasetMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class BaseGiftEvalDataset(ABC, IterableDataset):
    """
    Base dataset for GiftEval pretraining — Arrow IPC *stream* format.
    Subclasses should implement _build_batch().

    The dataset iterates indefinitely, yielding batches of size `batch_size`.
    Sampling is two-level and uniform:
      1. Pick a dataset (top-level subdirectory) uniformly at random.
      2. Pick a file from that dataset with probability proportional to its
         valid row count (so rows are sampled uniformly within the dataset).

    At iteration time each chosen file is fully deserialized via read_all()
    and released after the batch is built.  Use BaseFileFormatGiftEvalDataset
    if your data has been converted to Arrow IPC *file* format, which supports
    true random access and avoids loading the full file per batch.

    Args:
    -----
    root : str | Path
        Root directory containing the GiftEval pretraining data.
    batch_size : int
        Number of time series per batch.
    min_past : int
        Minimum number of non-padding time-steps required for a sample to be
        included.  Samples shorter than this are dropped. Default 1.
    dataset_names : list[str] | None
        Optional list of dataset names (subdirectory names under `root`) to
        include.  If None, all datasets found under `root` are included.
    arrow_paths : list[str | Path] | None
        Optional list of extra Arrow IPC *stream* files to include as
        additional datasets (e.g. kernelsynth-data.arrow).  Each file is
        treated as its own dataset entry and sampled with equal probability
        to the GiftEval datasets.  Default None.
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        min_past: int = 1,
        dataset_names: list[str] | None = None,
        arrow_paths: list[str | Path] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.min_past = min_past
        self._rng = None  # seeded in __iter__ from worker seed

        if not self.root.is_dir():
            raise RuntimeError(f"Root directory does not exist: {self.root}")

        # Group files by top-level subdirectory (= one GiftEval dataset)
        datasets_files: dict[str, list[Path]] = {}
        # For each dataset scan its files once to get row counts.
        # _datasets[i] = {
        #   "paths":   list[Path]
        #   "valid_rows": list[np.ndarray] — row indices passing min_past, per file
        #   "weights": np.ndarray — proportional to valid row count per file
        # }
        self._datasets: list[dict] = []
        for top in sorted(self.root.iterdir()):
            if not top.is_dir() or top.name.startswith("."):
                continue
            if dataset_names is not None and top.name not in dataset_names:
                continue
            files = sorted(top.rglob("data-*.arrow"))
            if files:
                datasets_files[top.name] = files

            paths: list[Path] = []
            valid_rows: list[np.ndarray] = []
            n_channels_list: list[int] = []
            for path in files:
                try:
                    mmap = pa.memory_map(str(path), "r")
                    table = pa_ipc.open_stream(mmap).read_all()
                    del mmap
                    col = table.column("target")
                    arr = col.combine_chunks() if col.num_chunks > 1 else col.chunk(0)
                    if pa.types.is_fixed_size_list(arr.type):
                        C = arr.type.list_size
                        inner_offsets = arr.values.offsets.to_numpy()
                        lengths = np.diff(inner_offsets[::C]).astype(np.int64)
                    else:
                        C = 1
                        lengths = np.diff(arr.offsets.to_numpy()).astype(np.int64)
                    valid = np.where(lengths >= min_past)[0].astype(np.int32)
                    if len(valid) > 0:
                        paths.append(path)
                        valid_rows.append(valid)
                        n_channels_list.append(C)
                except Exception:
                    continue
            if paths:
                counts = np.array(
                    [len(v) * c for v, c in zip(valid_rows, n_channels_list)],
                    dtype=np.float64,
                )
                self._datasets.append(
                    {
                        "name": top.name,
                        "paths": paths,
                        "valid_rows": valid_rows,
                        "n_channels": n_channels_list,
                        "weights": counts / counts.sum(),
                    }
                )

        if not datasets_files:
            raise RuntimeError(
                f"No 'data-*.arrow' files found under {self.root}. "
                "Check that `root` points to the GiftEvalPretrain directory."
            )

        if not self._datasets:
            raise RuntimeError(f"Could not open any Arrow files under {self.root}.")

        # ------------------------------------------------------------------ #
        # Extra Arrow IPC stream files (e.g. kernelsynth-data.arrow)
        # ------------------------------------------------------------------ #
        for raw_path in arrow_paths or []:
            path = Path(raw_path)
            try:
                mmap = pa.memory_map(str(path), "r")
                table = pa_ipc.open_stream(mmap).read_all()
                del mmap
                col = table.column("target")
                arr = col.combine_chunks() if col.num_chunks > 1 else col.chunk(0)
                if pa.types.is_fixed_size_list(arr.type):
                    C = arr.type.list_size
                    inner_offsets = arr.values.offsets.to_numpy()
                    lengths = np.diff(inner_offsets[::C]).astype(np.int64)
                else:
                    C = 1
                    lengths = np.diff(arr.offsets.to_numpy()).astype(np.int64)
                valid = np.where(lengths >= min_past)[0].astype(np.int32)
                if len(valid) > 0:
                    counts = np.array([len(valid) * C], dtype=np.float64)
                    self._datasets.append(
                        {
                            "name": path.stem,
                            "paths": [path],
                            "valid_rows": [valid],
                            "n_channels": [C],
                            "weights": counts / counts.sum(),
                        }
                    )
            except Exception:
                continue

    @abstractmethod
    def _build_batch(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Subclasses must implement _build_batch() "
            "to construct a batch from the given task indices."
        )

    def __len__(self):
        return sum(
            len(v) * c for ds in self._datasets for v, c in zip(ds["valid_rows"], ds["n_channels"])
        )

    def __iter__(self):
        n_datasets = len(self._datasets)
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        self._rng = np.random.default_rng(seed)
        rng = self._rng

        while True:
            ds = self._datasets[int(rng.integers(0, n_datasets))]

            sources: list[tuple[np.ndarray, np.ndarray, bool]] = []
            tables: list = []
            pending_src: list[int] = []
            pending_fr: list[int] = []
            pending_len: list[int] = []

            while len(pending_fr) < self.batch_size:
                file_idx = int(rng.choice(len(ds["paths"]), p=ds["weights"]))
                valid = ds["valid_rows"][file_idx]
                C = ds["n_channels"][file_idx]
                still_needed = self.batch_size - len(pending_fr)
                rows_needed = -(-still_needed // C)
                if len(valid) <= rows_needed:
                    sampled = valid
                else:
                    sampled = valid[rng.choice(len(valid), size=rows_needed, replace=False)]

                mmap = pa.memory_map(str(ds["paths"][file_idx]), "r")
                table = pa_ipc.open_stream(mmap).read_all()
                col = table.column("target")
                arr = col.combine_chunks() if col.num_chunks > 1 else col.chunk(0)

                if C > 1:
                    inner = arr.values
                    offsets = inner.offsets.to_numpy()
                    values = inner.values.to_numpy()
                else:
                    offsets = arr.offsets.to_numpy()
                    values = arr.values.to_numpy()

                is_f32 = values.dtype == np.float32
                src_idx = len(sources)
                sources.append((offsets, values, is_f32))
                tables.append((table, mmap, arr))

                for r in sampled:
                    if len(pending_fr) >= self.batch_size:
                        break
                    length = int(offsets[r * C + 1]) - int(offsets[r * C])
                    for c in range(C):
                        if len(pending_fr) >= self.batch_size:
                            break
                        pending_src.append(src_idx)
                        pending_fr.append(int(r) * C + c)
                        pending_len.append(length)

            yield self._build_batch(
                sources,
                pending_src[: self.batch_size],
                pending_fr[: self.batch_size],
                pending_len[: self.batch_size],
            )

            for table, mmap, arr in tables:
                mmap.close()

            del sources, tables, pending_src, pending_fr, pending_len


class BaseFileFormatGiftEvalDataset(ABC, IterableDataset):
    """
    Base dataset for GiftEval pretraining — Arrow IPC *file* (random-access) format.
    Subclasses should implement _build_batch().

    Requires data converted from stream format via scripts/stream_to_file.py.

    Advantages over BaseGiftEvalDataset (stream format):
    - Init: row counts come from the file footer — no data deserialized at all.
    - Iter: only the record batches containing the sampled rows are read.
      The OS pages in only those bytes, not the full file.

    Sampling is identical: uniform over datasets, then uniform over valid rows
    within the chosen dataset (implemented as proportional-to-count file pick).

    Args:
    -----
    root : str | Path
        Root directory containing the converted GiftEval data
        (e.g. GiftEvalPretrain_nostream).
    batch_size : int
        Number of time series per batch.
    min_past : int
        Minimum series length to include. Default 1.
    dataset_names : list[str] | None
        Subset of subdirectory names to use. Default: all.
    arrow_paths : list[str | Path] | None
        Optional list of extra Arrow IPC *file*-format files to include as
        additional datasets (e.g. kernelsynth-data.arrow).  Each file is
        treated as its own dataset entry and sampled with equal probability
        to the GiftEval datasets.  Default None.
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        min_past: int = 1,
        dataset_names: list[str] | None = None,
        arrow_paths: list[str | Path] | None = None,
        exclude_datasets: list[str] | None = None,
        num_files_per_batch: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.min_past = min_past
        self.num_files_per_batch = num_files_per_batch
        self._rng = None

        if self.num_files_per_batch is not None:
            self.max_per_file_batch = self.batch_size // self.num_files_per_batch
        else:
            self.max_per_file_batch = self.batch_size

        if not self.root.is_dir():
            raise RuntimeError(f"Root directory does not exist: {self.root}")

        self._datasets: list[dict] = []

        for top in sorted(self.root.iterdir()):
            if not top.is_dir() or top.name.startswith("."):
                continue
            if dataset_names is not None and top.name not in dataset_names:
                continue
            # Exclude if it contains any string in exclude_datasets
            if exclude_datasets is not None and any(excl in top.name for excl in exclude_datasets):
                continue

            paths: list[Path] = []
            # valid_rows[i] — int32 array of row indices passing min_past in file i
            valid_rows: list[np.ndarray] = []
            paths: list[Path] = []
            valid_rows: list[np.ndarray] = []
            batch_valid_rows: list[list[np.ndarray]] = []
            n_channels_list: list[int] = []

            for path in sorted(top.rglob("data-*.arrow")):
                try:
                    mmap = pa.memory_map(str(path), "r")
                    reader = pa_ipc.open_file(mmap)
                    n_batches = reader.num_record_batches
                    file_C: int | None = None
                    file_bvr: list[np.ndarray] = []
                    file_valid_count = 0

                    for b in range(n_batches):
                        batch = reader.get_batch(b)
                        col = batch.column("target")
                        if pa.types.is_fixed_size_list(col.type):
                            C = col.type.list_size
                            inner_offsets = col.values.offsets.to_numpy()
                            lengths = np.diff(inner_offsets[::C]).astype(np.int64)
                        else:
                            C = 1
                            lengths = np.diff(col.offsets.to_numpy()).astype(np.int64)
                        if file_C is None:
                            file_C = C
                        valid = np.where(lengths >= min_past)[0].astype(np.int32)
                        file_bvr.append(valid)
                        file_valid_count += len(valid)

                    del reader
                    del mmap

                    if file_valid_count > 0:
                        C = file_C or 1
                        paths.append(path)
                        # valid_rows[i] stores the concatenated valid local rows
                        # across all batches, used only for __len__
                        all_valid = np.concatenate(file_bvr) if file_bvr else np.empty(0, np.int32)
                        valid_rows.append(all_valid)
                        batch_valid_rows.append(file_bvr)
                        n_channels_list.append(C)
                except Exception:
                    continue

            if paths:
                counts = np.array(
                    [len(v) * c for v, c in zip(valid_rows, n_channels_list)],
                    dtype=np.float64,
                )
                non_empty_batches = [
                    [b for b, v in enumerate(bvr) if len(v) > 0]
                    for bvr in batch_valid_rows
                ]
                # Total valid univariate series in this dataset.
                total_series = int(counts.sum())
                # Per-dataset sampling weight: 0.01 for tiny datasets (< 100
                # series), 1.0 for everything else.  Sharded datasets such as
                # CMIP6 and ERA5 are detected by name prefix; their per-shard
                # weights are stored as 1.0 here and later rescaled so the
                # whole group collectively totals 1.0 (see post-loop below).
                dataset_weight = 0.01 if total_series < 100 else 1.0
                self._datasets.append(
                    {
                        "name": top.name,
                        "paths": paths,
                        "valid_rows": valid_rows,
                        "batch_valid_rows": batch_valid_rows,
                        "non_empty_batches": non_empty_batches,
                        "n_channels": n_channels_list,
                        "weights": counts / counts.sum(),
                        "dataset_weight": dataset_weight,
                    }
                )

        # ── Rescale sharded datasets (CMIP6, ERA5) so each group sums to 1.0
        for prefix in ("cmip6", "era5"):
            shard_idxs = [i for i, ds in enumerate(self._datasets) if ds["name"].startswith(prefix)]
            if shard_idxs:
                n_shards = len(shard_idxs)
                for i in shard_idxs:
                    self._datasets[i]["dataset_weight"] = 1.0 / n_shards

        # ------------------------------------------------------------------ #
        # Extra Arrow IPC file-format files (e.g. kernelsynth-data.arrow)
        # ------------------------------------------------------------------ #
        for raw_path in arrow_paths or []:
            path = Path(raw_path)
            try:
                mmap = pa.memory_map(str(path), "r")
                reader = pa_ipc.open_file(mmap)
                n_batches = reader.num_record_batches
                file_C: int | None = None
                file_bvr: list[np.ndarray] = []
                file_valid_count = 0

                for b in range(n_batches):
                    batch = reader.get_batch(b)
                    col = batch.column("target")
                    if pa.types.is_fixed_size_list(col.type):
                        C = col.type.list_size
                        inner_offsets = col.values.offsets.to_numpy()
                        lengths = np.diff(inner_offsets[::C]).astype(np.int64)
                    else:
                        C = 1
                        lengths = np.diff(col.offsets.to_numpy()).astype(np.int64)
                    if file_C is None:
                        file_C = C
                    valid = np.where(lengths >= min_past)[0].astype(np.int32)
                    file_bvr.append(valid)
                    file_valid_count += len(valid)

                del reader
                mmap.close()

                if file_valid_count > 0:
                    C = file_C or 1
                    all_valid = (
                        np.concatenate(file_bvr) if file_bvr else np.empty(0, np.int32)
                    )
                    counts = np.array([file_valid_count * C], dtype=np.float64)
                    non_empty_batches = [
                        [b for b, v in enumerate(file_bvr) if len(v) > 0]
                    ]
                    total_series = int(counts.sum())
                    self._datasets.append(
                        {
                            "name": path.stem,
                            "paths": [path],
                            "valid_rows": [all_valid],
                            "batch_valid_rows": [file_bvr],
                            "non_empty_batches": non_empty_batches,
                            "n_channels": [C],
                            "weights": counts / counts.sum(),
                            "dataset_weight": 0.01 if total_series < 100 else 1.0,
                        }
                    )
            except Exception:
                continue

        if not self._datasets:
            raise RuntimeError("Could not open any Arrow files.")

    @abstractmethod
    def _build_batch(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def sampling_weights(self) -> np.ndarray:
        """Normalized per-dataset sampling weights derived from dataset_weight.

        Weights are set at init time:
        - Datasets with < 100 total univariate series → 0.01
        - All other datasets → 1.0
        - CMIP6 / ERA5 shards are rescaled so each group collectively totals 1.0

        Returns a 1-D float64 array of length ``len(self._datasets)``
        whose values sum to 1.  Use this with ``MixtureDataset`` and
        ``strategy="given"`` to sample across real datasets by these weights.
        """
        w = np.array([ds["dataset_weight"] for ds in self._datasets], dtype=np.float64)
        return w / w.sum()

    def __len__(self):
        # Number of samples is sum of valid rows across all files, weighted by n_channels
        return sum(
            len(v) * c for ds in self._datasets for v, c in zip(ds["valid_rows"], ds["n_channels"])
        )

    def __iter__(self):
        n_datasets = len(self._datasets)
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        ds_weights = self.sampling_weights  # pre-computed; shape (n_datasets,)

        while True:

            sources: list[tuple[np.ndarray, np.ndarray, bool]] = []
            tables: list = []
            pending_src: list[int] = []
            pending_fr: list[int] = []
            pending_len: list[int] = []

            while len(pending_fr) < self.batch_size:
                ds = self._datasets[int(rng.choice(n_datasets, p=ds_weights))]
                
                file_idx = int(rng.choice(len(ds["paths"]), p=ds["weights"]))
                bvr: list[np.ndarray] = ds["batch_valid_rows"][file_idx]
                C = ds["n_channels"][file_idx]

                # Pick a record batch uniformly among those with valid rows
                non_empty = ds["non_empty_batches"][file_idx]
                batch_idx = non_empty[int(rng.integers(0, len(non_empty)))]
                valid_locals = bvr[batch_idx]

                still_needed = self.batch_size - len(pending_fr)
                rows_needed = min(-(-still_needed // C), self.max_per_file_batch)
                if len(valid_locals) <= rows_needed:
                    sampled_locals = valid_locals
                else:
                    sampled_locals = valid_locals[
                        rng.choice(len(valid_locals), size=rows_needed, replace=False)
                    ]

                mmap = pa.memory_map(str(ds["paths"][file_idx]), "r")
                reader = pa_ipc.open_file(mmap)
                rb = reader.get_batch(batch_idx)
                col = rb.column("target")
                if C > 1:
                    inner = col.values
                    offsets = inner.offsets.to_numpy()
                    values = inner.values.to_numpy()
                else:
                    offsets = col.offsets.to_numpy()
                    values = col.values.to_numpy()
                is_f32 = values.dtype == np.float32

                src_idx = len(sources)
                sources.append((offsets, values, is_f32))
                tables.append((reader, mmap))

                for local_r in sampled_locals:
                    if len(pending_fr) >= self.batch_size:
                        break
                    length = int(offsets[local_r * C + 1]) - int(offsets[local_r * C])
                    for c in range(C):
                        if len(pending_fr) >= self.batch_size:
                            break
                        pending_src.append(src_idx)
                        pending_fr.append(int(local_r) * C + c)
                        pending_len.append(length)

            yield self._build_batch(
                sources,
                pending_src[: self.batch_size],
                pending_fr[: self.batch_size],
                pending_len[: self.batch_size],
            )

            for reader, mmap in tables:
                mmap.close()

            del sources, tables, pending_src, pending_fr, pending_len


class BaseArrowFileDataset(ABC, IterableDataset):
    """
    Base dataset for memory-mapped Arrow files.
    Subclasses should implement _fetch_batch() to construct a batch from the given task indices.

    Args:
    -----
    arrow_path : str | Path
        Path to the Arrow IPC file containing the dataset.  Should have a "target" column.
    batch_size : int
        Number of time series per batch.
    min_past : int
        Minimum number of non-padding time-steps required for a sample to be
        included.  Samples shorter than this are dropped.  Default ``1``.
    mode : str | DatasetMode
        ``"train"`` (infinite random batches) or ``"validation"``
        (one-pass sequential batches).
    indices : np.ndarray | None
        Optional subset of row indices to use (e.g. for a train/val split).
        If None, all rows are used.
    """

    def __init__(
        self,
        arrow_path: str | Path,
        batch_size: int,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        min_past: int = 1,
        indices: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        mmap_source = pa.memory_map(arrow_path, "r")
        table = pa.ipc.open_file(mmap_source).read_all()
        self.batch_size = batch_size
        self.mode = mode
        self.min_past = min_past
        self.indices = indices

        # ------------------------------------------------------------------ #
        # Grab zero-copy numpy views of the Arrow "target" column buffers.
        # ------------------------------------------------------------------ #
        target_col = table.column("target")
        if target_col.num_chunks > 1:
            arr = target_col.combine_chunks()
        else:
            arr = target_col.chunk(0)

        self._offsets: np.ndarray = arr.offsets.to_numpy()  # int32 / int64
        self._values: np.ndarray = arr.values.to_numpy()  # float32 / float64
        self._values_f32: bool = self._values.dtype == np.float32

        # ------------------------------------------------------------------ #
        # Length-based filtering
        # ------------------------------------------------------------------ #
        all_lengths = np.diff(self._offsets).astype(np.int64)
        candidates = indices if indices is not None else np.arange(len(all_lengths))
        self.mode = DatasetMode(mode) if isinstance(mode, str) else mode

        if self.mode != DatasetMode.TEST:
            valid = candidates[all_lengths[candidates] >= min_past]
        else:
            valid = candidates

        if len(valid) == 0:
            raise ValueError(
                "Dataset is empty after filtering by min_past.  "
                "Provide longer time series or reduce min_past."
            )

        self._valid_indices: np.ndarray = valid
        self._valid_lengths: np.ndarray = all_lengths[valid]
        self._rng = None  # seeded in __iter__ from worker seed

    @abstractmethod
    def _fetch_batch(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement _fetch_batch() "
            "to construct a batch from the given task indices."
        )

    # ---------------------------------------------------------------------- #
    # Generators
    # ---------------------------------------------------------------------- #

    def _generate_train_batches(self) -> Iterator[dict[str, torch.Tensor]]:
        """Infinite random batches for training."""
        n = len(self._valid_indices)
        while True:
            task_indices = self._rng.integers(0, n, size=self.batch_size)
            yield self._fetch_batch(task_indices)

    def _generate_sequential_batches(self) -> Iterator[dict[str, torch.Tensor]]:
        """One-pass sequential batches for validation / test."""
        n = len(self._valid_indices)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            task_indices = np.arange(start, end)
            yield self._fetch_batch(task_indices)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        self._rng = np.random.default_rng(seed)
        if self.mode == DatasetMode.TRAIN:
            yield from self._generate_train_batches()
        else:
            yield from self._generate_sequential_batches()


class MixtureDataset(IterableDataset):
    """
    Dataset that samples from a mixture of multiple source datasets.
    Each batch is drawn from a single source dataset, with probabilities
    specified by `weights`.

    Args:
    -----
    datasets : list[IterableDataset]
        List of source datasets to sample from.
    strategy : str
        Sampling strategy.  Supported values:
        - "uniform": sample uniformly from all datasets.
        - "proportional": sample in proportion to dataset sizes (number of samples).
        - "given": use the provided `weights` argument for sampling probabilities.
    weights : list[float] | None
        Optional list of sampling weights for each dataset, used only if `strategy` is "given".  Should be non-negative and sum to 1.
    """

    def __init__(self, datasets: list[IterableDataset], strategy: str, weights: list[float] | None = None):
        super().__init__()
        self.datasets = datasets
        self.strategy = strategy
        self.weights = weights
        self._rng = None  # seeded in __iter__

        assert strategy in {"uniform", "proportional", "given"}, f"Unsupported strategy: {strategy}"

        # Accept any IterableDataset to allow synthetic on-the-fly sources.
        assert all(isinstance(ds, IterableDataset) for ds in self.datasets), (
            "All datasets must be torch.utils.data.IterableDataset instances"
        )

        self.datasets_lengths = [len(ds) for ds in self.datasets]
        self.weights = self._compute_weights()
        self.datasets_iters = None  # initialized in __iter__

    def _compute_weights(self) -> np.ndarray:
        if self.strategy == "uniform":
            weights = np.ones(len(self.datasets))
        elif self.strategy == "proportional":
            weights = np.array(self.datasets_lengths, dtype=np.float64)
        elif self.strategy == "given":
            if self.weights is None:
                raise ValueError("Weights must be provided when using 'given' strategy.")
            weights = np.array(self.weights, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        weights_sum = weights.sum()
        if weights_sum == 0:
            raise ValueError("All datasets have zero length, cannot compute sampling weights.")
        return weights / weights_sum

    def __len__(self):
        # Number of individual time series across all datasets.
        return sum(self.datasets_lengths)

    def __iter__(self):
        self.dataset_iters = [iter(ds) for ds in self.datasets]
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        self._rng = np.random.default_rng(seed)
        while True:
            ds_idx = int(self._rng.choice(len(self.datasets), p=self.weights))
            yield next(self.dataset_iters[ds_idx])
