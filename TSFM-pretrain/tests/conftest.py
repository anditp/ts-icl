from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHRONOS2_CONFIG_PATH = REPO_ROOT / "configs" / "test.yaml"
PATCHTST_FM_CONFIG_PATH = REPO_ROOT / "configs" / "test_patchtstfm_cpu.yaml"


def _load_yaml_config(path: pathlib.Path) -> SimpleNamespace:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


@pytest.fixture(scope="session")
def chronos2_test_config() -> SimpleNamespace:
    return _load_yaml_config(CHRONOS2_CONFIG_PATH)


@pytest.fixture(scope="session")
def patchtst_fm_test_config() -> SimpleNamespace:
    return _load_yaml_config(PATCHTST_FM_CONFIG_PATH)


@pytest.fixture(scope="session")
def dummy_chronos2_data() -> list[dict[str, np.ndarray]]:
    """50 random-walk univariate time series as list-of-dicts."""
    rng = np.random.default_rng(42)
    data = []
    for _ in range(50):
        length = rng.integers(100, 201)
        target = np.cumsum(rng.standard_normal(length)).astype(np.float32)
        data.append({"target": target})
    return data


@pytest.fixture(scope="session")
def dummy_arrow_table() -> pa.Table:
    """PyArrow table with 200 float32 series of length 120."""
    rng = np.random.default_rng(0)
    n_series, series_len = 200, 120
    arrays = [rng.standard_normal(series_len).astype(np.float32) for _ in range(n_series)]
    target_col = pa.array(arrays, type=pa.list_(pa.float32()))
    return pa.table({"target": target_col})


@pytest.fixture()
def arrow_ipc_file(tmp_path, dummy_arrow_table) -> pathlib.Path:
    """Arrow IPC *file* format (random-access) on disk."""
    path = tmp_path / "test_data.arrow"
    writer = pa_ipc.new_file(str(path), dummy_arrow_table.schema)
    writer.write_table(dummy_arrow_table)
    writer.close()
    return path


@pytest.fixture()
def gifteval_root(tmp_path, dummy_arrow_table) -> pathlib.Path:
    """Fake GiftEval directory with one sub-dataset in Arrow IPC *file* format."""
    ds_dir = tmp_path / "fake_dataset"
    ds_dir.mkdir(parents=True)
    path = ds_dir / "data-00000-of-00001.arrow"
    writer = pa_ipc.new_file(str(path), dummy_arrow_table.schema)
    writer.write_table(dummy_arrow_table)
    writer.close()
    return tmp_path
