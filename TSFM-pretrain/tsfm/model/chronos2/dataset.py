import math
from enum import Enum
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence, TypeAlias, cast

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    import datasets


TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


def left_pad_and_cat_2D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pads tensors in the list to the length of the longest tensor along the second axis, then concats
    these equal length tensors along the first axis.
    """
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            padding = torch.full(
                (n_variates, max_len - length),
                fill_value=torch.nan,
                device=tensor.device,
            )
            tensor = torch.cat([padding, tensor], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def validate_and_prepare_single_dict_task(
    task: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]],
    idx: int,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Validates and prepares a single dictionary task for Chronos2.

    Parameters
    ----------
    task
        A dictionary representing a time series that contains:
        - 'target' (required): a 1D or 2D 'torch.Tensor' or 'np.ndarray' of shape (history_length,)
        or (n_variates, history_length). Forecasts will be generated for items in 'target'.
        - 'past_covariates' (optional): a dict of past-only covariates or past values of known
        future covariates. The keys of the dict must be the names of the covariates and values must
        be a 1D 'torch.Tensor' or 'np.ndarray' with length equal to the 'history_length' of 'target'
        - 'future_covariates' (optional): a dict of future values of known future covariates.
        The keys of the dict must be the names of the covariates and values must be
        a 1D 'torch.Tensor' or 'np.ndarray' with length equal to 'prediction_length'.
        All keys in 'future_covariates' must also be present in 'past_covariates'.

    idx
        Index of this task in the list of tasks, used for error messages.

    prediction_length
        The length of the prediction horizon.

    Returns
    -------
    A tuple containing:
    - task_context_tensor:
        Concatenated tensor of target and past covariates of shape (group_size, history_length),
        the first 'task_n_targets' items along the first axis contain the target variables and the
        remaining items contain past-only covariates and past values of known future covariates.

    - task_future_covariates_tensor:
        Tensor of future covariates of shape (group_size, prediction_length).
        The last 'task_n_future_covariates' items along the first axis contain future covariates.
        All the remaining elements corresponding to target and past-only covariates are NaNs.

    - task_n_targets:
        Number of target variables in the task.

    - task_n_covariates:
        Number of covariates in the task.

    - task_n_future_covariates:
        Number of future covariates in the task.
    """

    allowed_keys = {"target", "past_covariates", "future_covariates"}

    # Validate keys
    keys = set(task.keys())
    if not keys.issubset(allowed_keys):
        raise ValueError(
            f"Found invalid keys in task at index {idx}. Allowed keys are {allowed_keys}, but found {keys}."
        )
    if "target" not in task:
        raise ValueError(f"Task at index {idx} is missing required key 'target'.")

    # Validate target
    task_target = task["target"]
    if isinstance(task_target, np.ndarray):
        task_target = torch.from_numpy(task_target)
    assert isinstance(task_target, torch.Tensor)
    if task_target.ndim > 2:
        raise ValueError(
            "When input is a list of dicts, the 'target' should either be a 1D with shape"
            "(history_length,) or 2D with shape (n_variates, history_length)."
            f"Found element at index {idx} with shape {tuple(task_target.shape)}."
        )
    history_length = task_target.shape[-1]
    task_target = task_target.view(-1, history_length)

    # Validate past_covariates
    cat_encoders: dict = {}
    task_past_covariates = task.get("past_covariates", {})
    if not isinstance(task_past_covariates, dict):
        raise ValueError(
            f"Found invalid type for 'past_covariates' in task at index {idx}."
            f'Expected a dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}},'
            f"but found {type(task_past_covariates)}."
        )

    # Gather keys and ensure future_covariates keys come last to match downstream assumptions
    task_covariates_keys = sorted(task_past_covariates.keys())

    task_future_covariates = task.get("future_covariates", {})
    if not isinstance(task_future_covariates, dict):
        raise ValueError(
            f"Found invalid type for 'future_covariates' in task at index {idx}."
            f'Expected a dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}},'
            f"but found {type(task_future_covariates)}."
        )
    task_future_covariates_keys = sorted(task_future_covariates.keys())
    if not set(task_future_covariates_keys).issubset(set(task_covariates_keys)):
        raise ValueError(
            f"Expected keys in 'future_covariates' to be a subset of keys in 'past_covariates',"
            f"but found {task_future_covariates_keys} in element at index {idx}."
        )

    # Create ordered keys: past-only first, then known-future
    task_past_only_keys = [
        k for k in task_covariates_keys if k not in task_future_covariates_keys
    ]  # past-only keys
    task_ordered_covariate_keys = task_past_only_keys + task_future_covariates_keys

    task_past_covariates_list: list[torch.Tensor] = []
    for key in task_ordered_covariate_keys:
        tensor = task_past_covariates[key]
        if isinstance(tensor, np.ndarray):
            # Apply encoding to categorical variates
            if not np.issubdtype(tensor.dtype, np.number):
                # Target encoding, if the target is 1D
                if task_target.shape[0] == 1:
                    cat_encoder = TargetEncoder(target_type="continuous", smooth=1.0)
                    X = tensor.astype(str).reshape(-1, 1)
                    y = task_target.view(-1).numpy()
                    mask = np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    cat_encoder.fit(X, y)

                # Ordinal encoding, if the target is > 1D
                else:
                    cat_encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    )
                    cat_encoder.fit(tensor.astype(str).reshape(-1, 1))
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(
                    tensor.shape
                )
                cat_encoders[key] = cat_encoder
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != history_length:
            raise ValueError(
                f"Individual 'past_covariates' tensors must be 1D with length equal to the length"
                f"of 'target' (= {history_length}), found {key} with shape {tuple(tensor.shape)} in"
                f"element at index {idx}."
            )
        task_past_covariates_list.append(tensor)
    task_past_covariates_tensor = (
        torch.stack(task_past_covariates_list, dim=0)
        if task_past_covariates_list
        else torch.zeros((0, history_length), device=task_target.device)
    )

    # Validate future_covariates
    task_future_covariates_list: list[torch.Tensor] = []
    for key in task_ordered_covariate_keys:
        # Future values of past-only covariates are filled with NaNs
        tensor = task_future_covariates.get(
            key, torch.full((prediction_length,), fill_value=torch.nan)
        )
        if isinstance(tensor, np.ndarray):
            # Apply encoding to categorical variates
            if not np.issubdtype(tensor.dtype, np.number):
                cat_encoder = cat_encoders[key]
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(
                    tensor.shape
                )
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != prediction_length:
            raise ValueError(
                f"Individual 'future_covariates' tensors must be 1D with length equal to"
                f"'prediction_length' (= {prediction_length}), found {key} with shape"
                f"{tuple(tensor.shape)} in element at index {idx}."
            )
        task_future_covariates_list.append(tensor)
    task_future_covariates_tensor = (
        torch.stack(task_future_covariates_list, dim=0)
        if task_future_covariates_list
        else torch.zeros((0, prediction_length), device=task_target.device)
    )

    # Future values of target series are filled with NaNs
    task_future_covariates_target_padding = torch.full(
        (task_target.shape[0], prediction_length),
        fill_value=torch.nan,
        device=task_target.device,
    )

    task_context_tensor = torch.cat([task_target, task_past_covariates_tensor], dim=0).to(
        dtype=torch.float32
    )
    task_future_covariates_tensor = torch.cat(
        [task_future_covariates_target_padding, task_future_covariates_tensor], dim=0
    ).to(dtype=torch.float32)
    task_n_targets = task_target.shape[0]
    task_n_covariates = task_past_covariates_tensor.shape[0]
    task_n_future_covariates = len(task_future_covariates_keys)

    return (
        task_context_tensor,
        task_future_covariates_tensor,
        task_n_targets,
        task_n_covariates,
        task_n_future_covariates,
    )


def convert_list_of_tensors_input_to_list_of_dicts_input(
    list_of_tensors: Sequence[TensorOrArray],
) -> list[dict[str, torch.Tensor]]:
    """Convert a list of tensors input format to a list of dictionaries input format.


    Parameters
    ----------
    list_of_tensors
        A sequence of tensors or numpy arrays, where each element represents a time series.
        Each element should be either 1-d with shape (history_length,) or 2-d with shape
        (n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 1-d or 2-d torch.Tensor of shape (history_length,) or (n_variates, history_length).
    """

    output: list[dict[str, torch.Tensor]] = []
    for idx, tensor in enumerate(list_of_tensors):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim > 2:
            raise ValueError(
                "When the input is a list of torch tensors or numpy arrays, the elements should"
                "either be 1-d with shape (history_length,) or 2-d with shape"
                "(n_variates, history_length)."
                f" Found element at index {idx} with shape {tuple(tensor.shape)}."
            )
        length = tensor.shape[-1]
        tensor = tensor.view(-1, length)

        output.append({"target": tensor})

    return output


def convert_tensor_input_to_list_of_dicts_input(
    tensor: TensorOrArray,
) -> list[dict[str, torch.Tensor]]:
    """
    Convert a tensor input format to a list of dictionaries input format.

    Parameters
    ----------
    tensor
        A tensor or numpy array representing multiple time series.
        Should be 3-d with shape (n_series, n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 2-d torch.Tensor of shape (n_variates, history_length).
    """

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim != 3:
        raise ValueError(
            "When the input is a torch tensor or numpy array, it should be 3-d with shape"
            "(n_series, n_variates, history_length)."
            f" Found shape: {tuple(tensor.shape)}."
        )

    output: list[dict[str, torch.Tensor]] = []
    n_series = len(tensor)
    for i in range(n_series):
        output.append({"target": tensor[i]})

    return output


class DatasetMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ZeroCopyArrowTaskList:
    """Zero-copy random access to Arrow list<float> columns.

    Arrow stores ``list<float>`` columns internally as two contiguous
    buffers:
        - **offsets** (int32/64): ``offsets[i]..offsets[i+1]`` spans row *i*
        - **values** (float32/64): the flat concatenation of every list

    By grabbing numpy views of these buffers (zero-copy when the table is
    memory-mapped via ``pa.memory_map``), random access to any row becomes
    a trivial slice ``values[offsets[i]:offsets[i+1]]``.
    """

    def __init__(
        self,
        table,  # pa.Table
        prediction_length: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        indices: np.ndarray | None = None,
    ):
        target_col = table.column("target")

        # Collapse to a single contiguous chunk so buffers are contiguous
        if target_col.num_chunks > 1:
            arr = target_col.combine_chunks()
        else:
            arr = target_col.chunk(0)

        # Zero-copy numpy views of Arrow's internal buffers.
        # When the table sits on a pa.memory_map these point directly into
        # the OS page cache — no RAM is consumed until pages are touched.
        self._offsets = arr.offsets.to_numpy()  # int32 or int64
        self._values = arr.values.to_numpy()  # float32 or float64
        self._values_are_f32 = self._values.dtype == np.float32

        all_lengths = np.diff(self._offsets)  # length of every row

        candidates = indices if indices is not None else np.arange(len(all_lengths))
        mode = DatasetMode(mode) if isinstance(mode, str) else mode

        if mode != DatasetMode.TEST:
            min_length = min_past + prediction_length
            self._valid_indices = candidates[all_lengths[candidates] >= min_length]
        else:
            self._valid_indices = candidates

        if len(self._valid_indices) == 0:
            raise ValueError(
                "Dataset empty after length filtering "
                "(need length >= min_past + prediction_length). "
                "Provide longer time series or reduce min_past / prediction_length."
            )

        self._prediction_length = prediction_length

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> tuple:
        arrow_idx = int(self._valid_indices[idx])
        start = self._offsets[arrow_idx]
        end = self._offsets[arrow_idx + 1]

        # Zero-copy slice from the (memory-mapped) values buffer.
        raw = self._values[start:end]

        # float64 → float32 conversion (single row ≈ microseconds, negligible)
        target_np = raw if self._values_are_f32 else raw.astype(np.float32)

        # Build the task tuple directly — identical contract to
        # validate_and_prepare_single_dict_task for the target-only case:
        #   (context_tensor, future_covariates_tensor,
        #    n_targets, n_covariates, n_future_covariates)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_np)).unsqueeze(
            0
        )  # (1, length)

        future_covariates = torch.full(
            (1, self._prediction_length),
            fill_value=torch.nan,
            dtype=torch.float32,
        )

        return (target_tensor, future_covariates, 1, 0, 0)


class ArrowBatchDataset(IterableDataset):
    """Vectorized batch-level Arrow dataset for pre-training.

    Instead of calling ``__getitem__`` N times per batch then stacking in
    ``default_collate`` (N Python calls + N tensor allocations + 1 stack),
    this dataset constructs entire batches directly from the memory-mapped
    Arrow buffers using vectorized numpy operations.

    For each batch:
    1. Sample ``batch_size`` random indices (train) or iterate sequentially (val)
    2. Sort indices within the batch → sequential mmap page access
    3. Vectorized gather of starts/ends from the offsets array
    4. Single pre-allocated numpy matrix → fill rows in a tight C-level loop
    5. One ``torch.from_numpy`` for the whole batch
    """

    def __init__(
        self,
        table,  # pa.Table — memory-mapped via pa.memory_map
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        indices: np.ndarray | None = None,
    ):
        super().__init__()
        target_col = table.column("target")
        if target_col.num_chunks > 1:
            arr = target_col.combine_chunks()
        else:
            arr = target_col.chunk(0)

        self._offsets = arr.offsets.to_numpy()
        self._values = arr.values.to_numpy()
        self._values_f32 = self._values.dtype == np.float32

        # Vectorized length computation + filtering
        all_lengths = np.diff(self._offsets).astype(np.int64)
        candidates = indices if indices is not None else np.arange(len(all_lengths))

        self.mode = DatasetMode(mode) if isinstance(mode, str) else mode

        if self.mode != DatasetMode.TEST:
            min_length = min_past + prediction_length
            valid = candidates[all_lengths[candidates] >= min_length]
        else:
            valid = candidates

        if len(valid) == 0:
            raise ValueError(
                "Dataset empty after length filtering "
                "(need length >= min_past + prediction_length)."
            )

        self._valid_indices = valid
        # Pre-compute lengths for valid indices (avoids recomputation per batch)
        self._valid_lengths = all_lengths[valid]

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.output_patch_size = output_patch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _fetch_batch(self, task_indices: np.ndarray) -> dict[str, torch.Tensor | int]:
        """Build a complete model-ready batch from dataset-local indices.

        Numpy dominant, using torch tensors only at the end.
        """
        bs = len(task_indices)
        ctx = self.context_length
        pred = self.prediction_length

        # Sort indices for sequential mmap page access
        order = np.argsort(task_indices)
        sorted_task_indices = task_indices[order]

        # Map dataset-local indices → Arrow row indices
        arrow_indices = self._valid_indices[sorted_task_indices]
        lengths = self._valid_lengths[sorted_task_indices]

        # Gather starts from offsets (vectorized)
        starts = self._offsets[arrow_indices]

        # Pre-allocate output buffers (NaN-filled for left-padding)
        context_buf = np.full((bs, ctx), np.nan, dtype=np.float32)
        future_buf = np.full((bs, pred), np.nan, dtype=np.float32)

        # Fill each row — this loop is irreducible (variable-length rows)
        # but sorted indices ensure sequential mmap reads
        for i_sorted in range(bs):
            L = int(lengths[i_sorted])
            s = int(starts[i_sorted])

            if self.mode == DatasetMode.TRAIN:
                # Random window: pick a split point
                max_split = L - pred
                split = np.random.randint(self.min_past, max_split + 1)
            elif self.mode == DatasetMode.VALIDATION:
                split = L - pred
            else:
                split = L

            # Context: take up to context_length before split
            ctx_start = max(0, split - ctx)
            ctx_len = split - ctx_start
            raw_ctx = self._values[s + ctx_start : s + split]
            if not self._values_f32:
                raw_ctx = raw_ctx.astype(np.float32)
            # Left-pad: place at the end of context buffer
            context_buf[i_sorted, ctx - ctx_len : ctx] = raw_ctx

            # Future target
            if self.mode != DatasetMode.TEST:
                raw_fut = self._values[s + split : s + split + pred]
                if not self._values_f32:
                    raw_fut = raw_fut.astype(np.float32)
                actual_pred = len(raw_fut)
                future_buf[i_sorted, :actual_pred] = raw_fut

        # Unsort back to original order
        inv_order = np.argsort(order)
        context_buf = context_buf[inv_order]
        future_buf = future_buf[inv_order]

        context_t = torch.from_numpy(context_buf.copy())
        future_target_t = torch.from_numpy(future_buf.copy())
        future_covariates_t = torch.full((bs, pred), fill_value=torch.nan, dtype=torch.float32)
        group_ids_t = torch.arange(bs, dtype=torch.long)

        return {
            "context": context_t,
            "future_target": future_target_t,
            "future_covariates": future_covariates_t,
            "group_ids": group_ids_t,
            "num_output_patches": self.num_output_patches,
        }

    def _generate_train_batches(self):
        """Infinite random batches for training."""
        n = len(self._valid_indices)
        while True:
            task_indices = np.random.randint(0, n, size=self.batch_size)
            yield self._fetch_batch(task_indices)

    def _generate_sequential_batches(self):
        """One-pass sequential batches for validation."""
        n = len(self._valid_indices)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            task_indices = np.arange(start, end)
            yield self._fetch_batch(task_indices)

    def __iter__(self) -> Iterator:
        if self.mode == DatasetMode.TRAIN:
            yield from self._generate_train_batches()
        else:
            yield from self._generate_sequential_batches()


class Chronos2Dataset(IterableDataset):
    """A dataset wrapper for Chronos-2 models.

    Parameters
    ----------
    inputs
        Time series data. Must be a list of dictionaries where each dictionary may have the following keys:
        - 'target' (required):
            a 1D or 2D 'torch.Tensor' or 'np.ndarray' of shape (history_length,) or (n_variates, history_length). Forecasts will be generated for items in 'target'.
        - 'past_covariates' (optional):
            a dict of past-only covariates or past values of known future covariates. The keys of the dict must be the names of the covariates and values must be a 1D 'torch.Tensor' or 'np.ndarray' with length equal to the 'history_length' of 'target'.
        - 'future_covariates' (optional):
            a dict of future values of known future covariates. The keys of the dict must be the names of the covariates and values must be a 1D 'torch.Tensor' or 'np.ndarray' with length equal to 'prediction_length'. All keys in 'future_covariates' must also be present in 'past_covariates'.
            Note: when the mode is set to TRAIN, the values inside `future_covariates` are not technically used for training the model; however, this key is used to infer which covariates are known into the future. Therefore, if your task contains known future covariates, make sure that this key exists in `inputs`. The values of individual future covariates may be set to `None` or an empty array.

    context_length
        The maximum context length used for training or inference.

    prediction_length
        The prediction horizon.

    batch_size
        The batch size for training the model. Note that the batch size here means the number of time series, including target(s) and covariates, that are input into the model. If your data has mutliple target and/or covariates, the effective number of time series tasks in a batch will be lower than this value.

    output_patch_size
        The output patch size of the model. This is used to compute the number of patches needed to cover 'prediction_length'.

    min_past
        The minimum number of time steps the context must have during training. All time series shorter than 'min_past + prediction_length' are filtered out, by default 1.

    mode
        'DatasetMode' governing whether to generate training, validation or test samples, by default 'train'.
    """

    def __init__(
        self,
        inputs: Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> None:
        super().__init__()
        assert mode in {DatasetMode.TRAIN, DatasetMode.VALIDATION, DatasetMode.TEST}, (
            f"Invalid mode: {mode}."
        )

        self.tasks = Chronos2Dataset._prepare_tasks(inputs, prediction_length, min_past, mode)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.output_patch_size = output_patch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.mode = mode
        self.min_past = min_past

    @staticmethod
    def _prepare_tasks(
        inputs: Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        prediction_length: int,
        min_past: int,
        mode: str | DatasetMode,
    ):
        tasks = []
        for idx, raw_task in enumerate(inputs):
            if mode != DatasetMode.TEST:
                raw_future_covariates = raw_task.get("future_covariates", {})
                raw_future_covariates = cast(dict[str, TensorOrArray | None], raw_future_covariates)
                if raw_future_covariates:
                    fixed_future_covariates = {}
                    for key, value in raw_future_covariates.items():
                        fixed_future_covariates[key] = (
                            np.full(prediction_length, np.nan)
                            if value is None or len(value) == 0
                            else value
                        )
                    raw_task = {
                        **raw_task,
                        "future_covariates": fixed_future_covariates,
                    }
            raw_task = cast(dict[str, TensorOrArray | Mapping[str, TensorOrArray]], raw_task)

            # Convert to a format compatible with the model's forward
            task = validate_and_prepare_single_dict_task(raw_task, idx, prediction_length)

            if mode != DatasetMode.TEST and task[0].shape[-1] < min_past + prediction_length:
                continue
            tasks.append(task)

        if len(tasks) == 0:
            raise ValueError(
                "The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length)."
                "Please provide longer time series or reduce 'min_past' or 'prediction_length'."
            )
        return tasks

    def _construct_slice(
        self, task_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        (
            task_past_tensor,
            task_future_tensor,
            task_n_targets,
            task_n_covariates,
            task_n_future_covariates,
        ) = self.tasks[task_idx]
        task_past_tensor, task_future_tensor = (
            task_past_tensor.clone(),
            task_future_tensor.clone(),
        )
        task_n_past_only_covariates = task_n_covariates - task_n_future_covariates

        full_length = task_past_tensor.shape[-1]

        if self.mode == DatasetMode.TRAIN:
            # Slice a random subsequence from the full series
            slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)
        elif self.mode == DatasetMode.VALIDATION:
            # Slice the last window for validation
            slice_idx = full_length - self.prediction_length
        else:
            # Slice the full series for prediction
            slice_idx = full_length

        if slice_idx >= self.context_length:
            # Slice series if it is longer than context_length
            task_context = task_past_tensor[:, slice_idx - self.context_length : slice_idx]
        else:
            task_context = task_past_tensor[:, :slice_idx]

        # In the TEST mode, we have no target available and the task_future_covariates can be directly used
        # In the TRAIN and VALIDATION modes, the target and task_future_covariates need to be constructed from
        # the task_context_tensor by slicing the appropriate indices which we do below

        if self.mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            # The first task_n_targets elements in task_context_tensor are the targets
            task_future_target = task_past_tensor[
                :, slice_idx : slice_idx + self.prediction_length
            ].clone()
            # Mask out all rows corresponding to covariates
            task_future_target[task_n_targets:, :] = torch.nan

            if task_n_future_covariates > 0:
                # The last task_n_future_covariates elements in task_context_tensor are the known covariates
                task_future_covariates = task_past_tensor[
                    -task_n_future_covariates:,
                    slice_idx : slice_idx + self.prediction_length,
                ]
            else:
                # Zero-length tensor for easy concatenation later
                task_future_covariates = torch.zeros((0, self.prediction_length))

            # The leading task_n_targets + task_n_past_only_covariates elements are masked because the target(s)
            # and past-only covariates are not known into the future
            task_future_covariates_padding = torch.full(
                (task_n_targets + task_n_past_only_covariates, self.prediction_length),
                fill_value=torch.nan,
            )
            task_future_covariates = torch.cat(
                [task_future_covariates_padding, task_future_covariates], dim=0
            )
        else:
            task_future_target = None
            task_future_covariates = task_future_tensor

        # task_context: (task_n_targets + task_n_covariates, min(context_length, history_length))
        # task_future_target: (task_n_targets + task_n_covariates, prediction_length), the future values of known future covariates
        # are ignored during loss computation
        # task_future_covariates: (task_n_targets + task_n_past_only_covariates + task_n_future_covariates, prediction_length),
        # the entries corresponding to targets and past-only covariates are NaNs

        return task_context, task_future_target, task_future_covariates, task_n_targets

    def _build_batch(
        self, task_indices: list[int]
    ) -> dict[str, torch.Tensor | int | list[tuple[int, int]] | None]:
        """Build a batch from the given task indices."""
        batch_context_tensor_list = []
        batch_future_target_tensor_list = []
        batch_future_covariates_tensor_list = []
        batch_group_ids_list = []
        target_idx_ranges: list[tuple[int, int]] = []

        target_start_idx = 0
        for group_id, task_idx in enumerate(task_indices):
            task_context, task_future_target, task_future_covariates, task_n_targets = (
                self._construct_slice(task_idx)
            )

            group_size = task_context.shape[0]
            task_group_ids = torch.full((group_size,), fill_value=group_id)
            batch_context_tensor_list.append(task_context)
            batch_future_target_tensor_list.append(task_future_target)
            batch_future_covariates_tensor_list.append(task_future_covariates)
            batch_group_ids_list.append(task_group_ids)
            target_idx_ranges.append((target_start_idx, target_start_idx + task_n_targets))
            target_start_idx += group_size

        return {
            "context": left_pad_and_cat_2D(batch_context_tensor_list),
            "future_target": None
            if self.mode == DatasetMode.TEST
            else torch.cat(cast(list[torch.Tensor], batch_future_target_tensor_list), dim=0),
            "future_covariates": torch.cat(batch_future_covariates_tensor_list, dim=0),
            "group_ids": torch.cat(batch_group_ids_list, dim=0),
            "num_output_patches": self.num_output_patches,
            "target_idx_ranges": target_idx_ranges,
        }

    def _generate_train_batches(self):
        while True:
            current_batch_size = 0
            task_indices = []

            while current_batch_size < self.batch_size:
                task_idx = np.random.randint(len(self.tasks))
                task_indices.append(task_idx)
                current_batch_size += self.tasks[task_idx][0].shape[0]

            yield self._build_batch(task_indices)

    def _generate_sequential_batches(self):
        task_idx = 0
        while task_idx < len(self.tasks):
            current_batch_size = 0
            task_indices = []

            while task_idx < len(self.tasks) and current_batch_size < self.batch_size:
                task_indices.append(task_idx)
                current_batch_size += self.tasks[task_idx][0].shape[0]
                task_idx += 1

            yield self._build_batch(task_indices)

    def __iter__(self) -> Iterator:
        """
        Generate batches of data for the Chronos-2 model. In training mode, this iterator is infinite.

        Yields
        ------
        dict
            A dictionary containing:
            - context: torch.Tensor of shape (batch_size, context_length) containing input sequences
            - future_target: torch.Tensor of shape (batch_size, prediction_length) containing future target sequences, None in TEST mode
            - future_covariates: torch.Tensor of shape (batch_size, prediction_length) containing known future covariates
            - group_ids: torch.Tensor of shape (batch_size,) containing the group ID for each sequence
            - num_output_patches: int indicating number of patches the model should output to cover prediction_length
            - target_idx_ranges: (only in TEST mode) list of tuples indicating the start & end indices of targets in context
        """
        if self.mode == DatasetMode.TRAIN:
            for batch in self._generate_train_batches():
                batch.pop("target_idx_ranges")
                yield batch
        elif self.mode == DatasetMode.VALIDATION:
            for batch in self._generate_sequential_batches():
                batch.pop("target_idx_ranges")
                yield batch
        else:
            yield from self._generate_sequential_batches()

    @classmethod
    def convert_inputs(
        cls,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> "Chronos2Dataset":
        """Convert from different input formats to a Chronos2Dataset."""
        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            inputs = convert_tensor_input_to_list_of_dicts_input(inputs)
        elif isinstance(inputs, list) and all(
            [isinstance(x, (torch.Tensor, np.ndarray)) for x in inputs]
        ):
            inputs = cast(list[TensorOrArray], inputs)
            inputs = convert_list_of_tensors_input_to_list_of_dicts_input(inputs)
        elif isinstance(inputs, list) and all([isinstance(x, dict) for x in inputs]):
            pass
        else:
            raise ValueError("Unexpected inputs format")

        inputs = cast(list[dict[str, TensorOrArray | dict[str, TensorOrArray]]], inputs)

        return cls(
            inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode=mode,
        )

    @classmethod
    def from_arrow(
        cls,
        table,  # pa.Table — ideally memory-mapped
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        indices: np.ndarray | None = None,
    ) -> "ArrowBatchDataset":
        """Create a vectorized Arrow dataset for pre-training.

        Returns an ``ArrowBatchDataset`` that yields complete model-ready
        batches directly from memory-mapped Arrow buffers, bypassing all
        per-sample Python overhead.

        Parameters
        ----------
        table : pyarrow.Table
            Arrow table (ideally via ``pa.memory_map``) containing at minimum
            a ``target`` column of list<float> type.
        indices : np.ndarray, optional
            Subset of row indices to use (e.g. for train/val split).
        """
        return ArrowBatchDataset(
            table=table,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode=mode,
            indices=indices,
        )
