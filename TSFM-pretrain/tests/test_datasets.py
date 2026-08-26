from __future__ import annotations

import math

import torch

from tsfm.model.chronos2.dataset import ArrowBatchDataset, Chronos2Dataset, DatasetMode
from tsfm.model.patchtst_fm.dataset import (
    PatchTSTFMArrowDataset,
    PatchTSTFMGiftEval,
    PatchTSTFMPatternRandomWalkDataset,
)

CHRONOS2_EXPECTED_KEYS = {
    "context",
    "future_target",
    "future_covariates",
    "group_ids",
    "num_output_patches",
}
PATCHTST_FM_EXPECTED_KEYS = {"inputs", "pred_mask", "miss_mask", "pad_mask"}


def test_chronos2_dataset_keys_and_shapes(dummy_chronos2_data, chronos2_test_config):
    cfg = chronos2_test_config
    dataset = Chronos2Dataset(inputs=dummy_chronos2_data, mode="train", **cfg.data)
    batch = next(iter(dataset))

    assert set(batch.keys()) == CHRONOS2_EXPECTED_KEYS
    assert batch["context"].ndim == 2
    assert batch["context"].shape[0] >= cfg.data["batch_size"]
    assert batch["context"].shape[1] == cfg.data["context_length"]
    assert batch["future_target"].shape[0] == batch["context"].shape[0]
    assert batch["future_target"].shape[1] == cfg.data["prediction_length"]
    assert batch["group_ids"].shape == (batch["context"].shape[0],)
    assert isinstance(batch["num_output_patches"], int)


def test_arrow_batch_dataset_keys_and_shapes(dummy_arrow_table):
    ctx_len, pred_len, bs, patch = 64, 16, 32, 8
    dataset = ArrowBatchDataset(
        table=dummy_arrow_table,
        context_length=ctx_len,
        prediction_length=pred_len,
        batch_size=bs,
        output_patch_size=patch,
        min_past=3,
        mode=DatasetMode.TRAIN,
    )
    batch = next(iter(dataset))

    assert set(batch.keys()) == CHRONOS2_EXPECTED_KEYS
    assert batch["context"].shape == (bs, ctx_len)
    assert batch["future_target"].shape == (bs, pred_len)
    assert batch["future_covariates"].shape == (bs, pred_len)
    assert batch["group_ids"].shape == (bs,)
    assert batch["num_output_patches"] == math.ceil(pred_len / patch)


def test_patchtst_fm_arrow_dataset_keys_and_shapes(arrow_ipc_file):
    ctx_len, bs = 64, 8
    dataset = PatchTSTFMArrowDataset(
        arrow_path=str(arrow_ipc_file),
        context_length=ctx_len,
        d_patch=16,
        batch_size=bs,
    )
    batch = next(iter(dataset))

    assert set(batch.keys()) == PATCHTST_FM_EXPECTED_KEYS
    for key in PATCHTST_FM_EXPECTED_KEYS:
        assert batch[key].shape == (bs, ctx_len), f"{key} has wrong shape"
        assert batch[key].dtype == torch.float32, f"{key} has wrong dtype"


def test_patchtst_fm_gifteval_keys_and_shapes(gifteval_root):
    ctx_len, bs = 64, 8
    dataset = PatchTSTFMGiftEval(
        root=str(gifteval_root),
        context_length=ctx_len,
        d_patch=16,
        batch_size=bs,
    )
    batch = next(iter(dataset))

    assert set(batch.keys()) == PATCHTST_FM_EXPECTED_KEYS
    for key in PATCHTST_FM_EXPECTED_KEYS:
        assert batch[key].shape == (bs, ctx_len), f"{key} has wrong shape"
        assert batch[key].dtype == torch.float32, f"{key} has wrong dtype"


def test_patchtst_fm_gifteval_drs_applies_downsampling(gifteval_root):
    """With fixed DRS=8 and fixed context length 64, downsample before pad.

    The fixture series length is 120, so after DRS (stride 8) effective length
    is 15 and pad_mask should contain exactly 49 ones per sample.
    """
    ctx_len, bs = 64, 8
    dataset = PatchTSTFMGiftEval(
        root=str(gifteval_root),
        context_length=ctx_len,
        d_patch=16,
        batch_size=bs,
        random_context_length=False,
        drs_factors=[8],
    )
    batch = next(iter(dataset))

    pad_counts = batch["pad_mask"].sum(dim=1)
    expected_pad = torch.full((bs,), 49.0)
    assert torch.allclose(pad_counts, expected_pad), (
        f"Expected 49 padded positions per sample with DRS=8, got {pad_counts.tolist()}"
    )


def test_patchtst_fm_pattern_random_walk_keys_and_shapes():
    ctx_len, bs = 64, 8
    dataset = PatchTSTFMPatternRandomWalkDataset(
        context_length=ctx_len,
        d_patch=16,
        batch_size=bs,
        random_context_length=False,
        motif_len_min=8,
        motif_len_max=16,
    )
    batch = next(iter(dataset))

    assert set(batch.keys()) == PATCHTST_FM_EXPECTED_KEYS
    for key in PATCHTST_FM_EXPECTED_KEYS:
        assert batch[key].shape == (bs, ctx_len), f"{key} has wrong shape"
        assert batch[key].dtype == torch.float32, f"{key} has wrong dtype"
