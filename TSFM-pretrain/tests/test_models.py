from __future__ import annotations

from tsfm.model.chronos2.chronos2 import Chronos2Model
from tsfm.model.chronos2.config import Chronos2CoreConfig
from tsfm.model.patchtst_fm import (
    PatchTSTFMConfig,
    PatchTSTFMForPrediction,
    PatchTSTFMForPretraining,
)


def test_chronos2_model_instantiation(chronos2_test_config):
    cfg = chronos2_test_config
    core_config = Chronos2CoreConfig(**cfg.model)
    model = Chronos2Model(core_config)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert model.config.d_model == cfg.model["d_model"]
    assert model.config.num_layers == cfg.model["num_layers"]
    assert model.config.num_heads == cfg.model["num_heads"]


def test_patchtst_fm_pretraining_instantiation(patchtst_fm_test_config):
    cfg = patchtst_fm_test_config
    model_cfg = PatchTSTFMConfig(**cfg.model)
    model = PatchTSTFMForPretraining(model_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert model.config.d_model == cfg.model["d_model"]
    assert model.config.n_layer == cfg.model["n_layer"]


def test_patchtst_fm_prediction_instantiation(patchtst_fm_test_config):
    cfg = patchtst_fm_test_config
    model_cfg = PatchTSTFMConfig(**cfg.model)
    pretrain_model = PatchTSTFMForPretraining(model_cfg)
    predict_model = PatchTSTFMForPrediction(model_cfg)

    pretrain_params = sum(p.numel() for p in pretrain_model.parameters())
    predict_params = sum(p.numel() for p in predict_model.parameters())
    assert predict_params > 0
    assert predict_params == pretrain_params
