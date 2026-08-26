"""Evaluate on Gift-Eval using a checkpoint from training.
For PatchTST-FM.

Loads weights directly from a .ckpt file into PatchTSTFMForPrediction
without requiring conversion to safetensors / HuggingFace format.

Usage example:
    python scripts/checkpoint_evaluation.py \
        --ckpt-path /path/to/checkpoint/step-N.ckpt \
        --output-dir /path/to/output/dir \
        --dataset-properties /path/to/dataset_properties.json

    # Run on specific datasets only:
    python scripts/checkpoint_evaluation.py \
        --ckpt-path /path/to/step-50000.ckpt \
        --output-dir results/my_run \
        --datasets m4_weekly electricity/H
"""

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tsfm.model.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPrediction

logger = logging.getLogger(__name__)

# Suppress noisy gluonts warning about mean prediction
_gts_logger = logging.getLogger("gluonts.model.forecast")


class _WarningFilter(logging.Filter):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def filter(self, record):
        return self.text not in record.getMessage()


_gts_logger.addFilter(_WarningFilter("The mean prediction is not stored in the forecast data"))

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T "
    "M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
]


def load_model(ckpt_path: str, device: str) -> PatchTSTFMForPrediction:
    """Load PatchTSTFMForPrediction directly from a training checkpoint.

    Handles:
    - Raw .ckpt files from the Trainer (direct state_dict loading, no HF conversion)
    - HuggingFace directories (config.json + model.safetensors, via from_pretrained)

    For raw checkpoints, the config dict stored in checkpoint["config"]
    is used to build a PatchTSTFMConfig, and weights are loaded via
    load_state_dict — no safetensors intermediate needed.
    """
    ckpt = Path(ckpt_path)

    if ckpt.is_dir() and (ckpt / "config.json").exists():
        logger.info("Loading from HuggingFace directory: %s", ckpt)
        return PatchTSTFMForPrediction.from_pretrained(str(ckpt), device_map=device)

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading raw checkpoint: %s", ckpt)
    checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    config_data = checkpoint["config"]
    if hasattr(config_data, "__dict__"):
        config_data = vars(config_data)
    config_data = dict(config_data)  # shallow copy

    # PatchTSTFMConfig.attribute_map maps hidden_size → d_model and
    # num_hidden_layers → n_layer.
    # Explicitly propagate the mapped aliases so they stay in sync.
    if "d_model" in config_data:
        config_data.setdefault("hidden_size", config_data["d_model"])
    if "n_layer" in config_data:
        config_data.setdefault("num_hidden_layers", config_data["n_layer"])

    config = PatchTSTFMConfig(**config_data)
    logger.info(
        "Config: d_model=%d, n_head=%d, n_layer=%d, context_length=%d, num_quantile=%d",
        config.d_model,
        config.n_head,
        config.n_layer,
        config.context_length,
        config.num_quantile,
    )

    model = PatchTSTFMForPrediction(config)

    # Clean state_dict keys:
    # strip _orig_mod. (torch.compile) and module. (DDP)
    raw_sd = checkpoint["state_dict"]
    clean_sd = OrderedDict()
    for k, v in raw_sd.items():
        key = k.replace("module.", "")
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        clean_sd[key] = v

    model.load_state_dict(clean_sd)
    model = model.to(device)
    model.eval()

    step = checkpoint.get("curr_step", "?")
    logger.info(
        "Loaded checkpoint (step %s) — %s params",
        step,
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
    )
    return model


class PatchTSTFMPredictor:
    def __init__(
        self,
        model: PatchTSTFMForPrediction,
        prediction_length: int,
        quantile_levels: List[float] = QUANTILE_LEVELS,
    ):
        self.model = model
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels

    @torch.no_grad()
    def predict(self, test_data_input, batch_size: int = 512) -> List[QuantileForecast]:
        items = list(test_data_input)
        forecasts: List[QuantileForecast] = []

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]

            inputs = []
            for item in batch:
                t = torch.as_tensor(
                    np.asarray(item["target"], dtype=np.float64),
                    dtype=torch.float32,
                )
                if t.ndim > 1:
                    t = t.squeeze()
                if torch.isnan(t).any():
                    fill = t.nanmean()
                    if torch.isnan(fill):
                        fill = torch.tensor(0.0)
                    t = torch.nan_to_num(t, nan=fill.item())
                inputs.append(t)

            try:
                output = self.model(
                    inputs=inputs,
                    prediction_length=self.prediction_length,
                    quantile_levels=self.quantile_levels,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at batch_size=%d, retrying with half", len(inputs))
                mid = len(batch) // 2
                forecasts.extend(self.predict(batch[:mid], batch_size=mid))
                forecasts.extend(self.predict(batch[mid:], batch_size=len(batch) - mid))
                continue

            preds = output.quantile_predictions.float().cpu().numpy()
            n_nan = np.isnan(preds).sum()
            if n_nan > 0:
                logger.warning(
                    "batch starting at %d: %d NaN forecast values replaced with 0",
                    start,
                    n_nan,
                )
                preds = np.nan_to_num(preds, nan=0.0)

            for i, item in enumerate(batch):
                forecast_start = item["start"] + len(item["target"])
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=preds[i],
                        forecast_keys=list(map(str, self.quantile_levels)),
                        start_date=forecast_start,
                    )
                )

        return forecasts


def resolve_ds_config(
    ds_name: str,
    term: str,
    dataset: Dataset,
    dataset_properties: Optional[dict],
) -> str:
    """Build canonical config string ``ds_key/freq/term``."""
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0].lower()
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        if dataset_properties and ds_key in dataset_properties:
            ds_freq = dataset_properties[ds_key]["frequency"]
        elif dataset_properties and PRETTY_NAMES.get(ds_key, ds_key) in dataset_properties:
            ds_freq = dataset_properties[PRETTY_NAMES[ds_key]]["frequency"]
        else:
            ds_freq = str(dataset.freq)
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    return f"{ds_key}/{ds_freq}/{term}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PatchTST-FM on the GIFT-Eval benchmark "
        "directly from a training checkpoint."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to checkpoint (.ckpt file or HF directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write results CSV into",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PatchTST-FM",
        help="Model name label for the results CSV (default: PatchTST-FM)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: auto-detect cuda > mps > cpu)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Override dataset list (e.g. m4_weekly electricity/H)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="+",
        default=None,
        help="Terms to evaluate (default: short + medium/long for eligible datasets)",
    )
    parser.add_argument(
        "--dataset-properties",
        type=str,
        default=None,
        help="Path to dataset_properties.json (optional, for config naming)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Device
    device = args.device
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
        )
    logger.info("Device: %s", device)

    model = load_model(args.ckpt_path, device)

    if args.terms is not None:
        terms_to_try = args.terms
        apply_med_long_filter = False
    else:
        terms_to_try = ("short", "medium", "long")
        apply_med_long_filter = True

    dataset_properties = None
    if args.dataset_properties:
        with open(args.dataset_properties) as f:
            dataset_properties = json.load(f)

    short_set = set(SHORT_DATASETS.split())
    med_long_set = set(MED_LONG_DATASETS.split())

    if args.datasets:
        all_datasets = args.datasets
    else:
        all_datasets = sorted(short_set | med_long_set)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds_num, ds_name in enumerate(all_datasets):
        logger.info("=== Dataset %d/%d: %s ===", ds_num + 1, len(all_datasets), ds_name)

        for term in terms_to_try:
            if apply_med_long_filter:
                if term in ("medium", "long") and ds_name not in med_long_set:
                    continue

            try:
                raw_ds = Dataset(name=ds_name, term=term, to_univariate=False)
            except Exception as exc:
                logger.warning("  skip %s/%s: %s", ds_name, term, exc)
                continue

            to_univariate = raw_ds.target_dim > 1
            dataset = (
                Dataset(name=ds_name, term=term, to_univariate=True) if to_univariate else raw_ds
            )

            ds_config = resolve_ds_config(ds_name, term, dataset, dataset_properties)
            logger.info(
                "  config=%-30s  pred_len=%d  n_test=%d",
                ds_config,
                dataset.prediction_length,
                len(dataset.test_data),
            )

            predictor = PatchTSTFMPredictor(
                model=model,
                prediction_length=dataset.prediction_length,
            )

            forecasts = predictor.predict(dataset.test_data.input, batch_size=args.batch_size)

            season_length = get_seasonality(dataset.freq)
            res = (
                evaluate_forecasts(
                    forecasts,
                    test_data=dataset.test_data,
                    metrics=METRICS,
                    axis=None,
                    batch_size=1024,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

            row = {"dataset": ds_config, "model": args.model_name}
            row.update({f"eval_metrics/{k}": v for k, v in res[0].items()})

            if dataset_properties:
                ds_key_clean = PRETTY_NAMES.get(
                    ds_name.split("/")[0].lower(),
                    ds_name.split("/")[0].lower(),
                )
                props = dataset_properties.get(ds_key_clean, {})
                row["domain"] = props.get("domain", "")
                row["num_variates"] = props.get("num_variates", "")

            all_results.append(row)
            logger.info(
                "  MASE=%.4f  NRMSE=%.4f  WQL=%.4f",
                row.get("eval_metrics/MASE[0.5]", float("nan")),
                row.get("eval_metrics/NRMSE[mean]", float("nan")),
                row.get("eval_metrics/mean_weighted_sum_quantile_loss", float("nan")),
            )

    results_df = pd.DataFrame(all_results).sort_values(by="dataset")
    csv_path = output_dir / "all_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
