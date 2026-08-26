from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from tsfm.model.chronos2.chronos2 import Chronos2Model
from tsfm.model.chronos2.config import Chronos2CoreConfig
from tsfm.model.chronos2.dataset import ArrowBatchDataset, Chronos2Dataset, DatasetMode
from tsfm.model.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPretraining
from tsfm.model.patchtst_fm.dataset import PatchTSTFMArrowDataset
from tsfm.train.trainer import PatchTSTFMTrainer, Trainer, worker_init_fn
from tsfm.train.trainer import split_scion_compatible_params


def test_split_scion_compatible_params_by_ndim():
    named_params = [
        ("linear.weight", torch.nn.Parameter(torch.randn(4, 8))),
        ("linear.bias", torch.nn.Parameter(torch.randn(8))),
        ("scalar", torch.nn.Parameter(torch.tensor(1.0))),
        ("frozen.weight", torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)),
    ]

    scion_named, fallback_named = split_scion_compatible_params(named_params, min_ndim=2)

    assert [name for name, _ in scion_named] == ["linear.weight"]
    assert {name for name, _ in fallback_named} == {"linear.bias", "scalar"}

    scion_named_1d, fallback_named_1d = split_scion_compatible_params(named_params, min_ndim=1)
    assert {name for name, _ in scion_named_1d} == {"linear.weight", "linear.bias"}
    assert {name for name, _ in fallback_named_1d} == {"scalar"}


@pytest.mark.slow
def test_chronos2_training_loop(chronos2_test_config, dummy_chronos2_data, tmp_path):
    cfg = copy.deepcopy(chronos2_test_config)
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.resume_training = False

    core_config = Chronos2CoreConfig(**cfg.model)
    model = Chronos2Model(core_config)

    n_val = int(len(dummy_chronos2_data) * cfg.val_split)
    train_data = dummy_chronos2_data[n_val:]
    val_data = dummy_chronos2_data[:n_val]

    train_ds = Chronos2Dataset(inputs=train_data, mode="train", **cfg.data)
    val_ds = Chronos2Dataset(inputs=val_data, mode="validation", **cfg.data)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)

    trainer = Trainer(cfg, model, train_loader, val_dataloader=val_loader)
    trainer.train()

    assert trainer.curr_step == cfg.max_steps


@pytest.mark.slow
def test_patchtst_fm_training_loop(patchtst_fm_test_config, arrow_ipc_file, tmp_path):
    cfg = copy.deepcopy(patchtst_fm_test_config)
    cfg.checkpoint_dir = str(tmp_path / "ckpts_ptst")
    cfg.resume_training = False

    model_cfg = PatchTSTFMConfig(**cfg.model)
    model = PatchTSTFMForPretraining(model_cfg)

    dataset = PatchTSTFMArrowDataset(
        arrow_path=str(arrow_ipc_file),
        context_length=cfg.data["context_length"],
        d_patch=cfg.data["d_patch"],
        batch_size=cfg.micro_batch_size,
        mask_ratio=cfg.data.get("mask_ratio", 0.4),
        n_cpm=cfg.data.get("n_cpm", 2),
        min_past=cfg.data.get("min_past", 16),
    )
    train_loader = DataLoader(dataset, batch_size=None, num_workers=0)

    trainer = PatchTSTFMTrainer(cfg, model, train_loader)
    trainer.train()

    assert trainer.curr_step == cfg.max_steps


@pytest.mark.slow
def test_worker_init_fn_diverse_batches(dummy_arrow_table):
    dataset_kwargs = dict(
        table=dummy_arrow_table,
        context_length=64,
        prediction_length=16,
        batch_size=32,
        output_patch_size=8,
        min_past=3,
        mode=DatasetMode.TRAIN,
    )
    num_workers = 2
    n_batches = num_workers * 4

    ds = ArrowBatchDataset(**dataset_kwargs)
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
        multiprocessing_context="fork",
    )
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(batch["context"])

    n_unique = len({b.numpy().tobytes() for b in batches})
    assert n_unique == len(batches)


@pytest.mark.slow
def test_checkpoint_save_load_roundtrip(chronos2_test_config, dummy_chronos2_data, tmp_path):
    cfg = copy.deepcopy(chronos2_test_config)
    cfg.checkpoint_dir = str(tmp_path / "ckpts_roundtrip")
    cfg.resume_training = False
    cfg.max_steps = 2

    core_config = Chronos2CoreConfig(**cfg.model)
    model = Chronos2Model(core_config)

    train_ds = Chronos2Dataset(inputs=dummy_chronos2_data, mode="train", **cfg.data)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)

    trainer = Trainer(cfg, model, train_loader)
    trainer.train()
    trainer.save_checkpoint("step-2.ckpt")

    # Snapshot trained weights
    trained_state = {k: v.clone() for k, v in trainer.raw_model.state_dict().items()}

    # Load into a fresh model
    ckpt = torch.load(
        str(tmp_path / "ckpts_roundtrip" / "step-2.ckpt"),
        map_location="cpu",
        weights_only=False,
    )
    fresh_model = Chronos2Model(core_config)
    fresh_model.load_state_dict(ckpt["state_dict"])

    assert ckpt["curr_step"] == 2
    for key in trained_state:
        assert torch.equal(trained_state[key], fresh_model.state_dict()[key]), (
            f"Parameter {key} mismatch after checkpoint roundtrip"
        )


@pytest.mark.slow
def test_gradient_accumulation(chronos2_test_config, dummy_chronos2_data, tmp_path):
    cfg = copy.deepcopy(chronos2_test_config)
    cfg.checkpoint_dir = str(tmp_path / "ckpts_ga")
    cfg.resume_training = False
    cfg.gradient_accumulation_steps = 2
    cfg.micro_batch_size = 64
    cfg.max_steps = 2

    core_config = Chronos2CoreConfig(**cfg.model)
    model = Chronos2Model(core_config)

    train_ds = Chronos2Dataset(inputs=dummy_chronos2_data, mode="train", **cfg.data)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)

    trainer = Trainer(cfg, model, train_loader)
    assert trainer.batch_size == 64 * 2

    trainer.train()
    assert trainer.curr_step == 2
