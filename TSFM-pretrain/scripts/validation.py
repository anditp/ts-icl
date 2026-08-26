import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from chronos.chronos2 import Chronos2Pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the validation data.",
        required=True,
        nargs="+",
    )
    args = parser.parse_args()

    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.1
    NP_SEED = 0
    PREDICTION_LEN = 32
    ZERO_SHOT = True

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    arrow_path = "./kernelsynth-data.arrow"
    table = pa.ipc.open_file(arrow_path).read_all()
    records = table.to_pylist()
    dict_inputs = []
    for rec in records:
        target = np.asarray(rec["target"], dtype=np.float32)
        dict_inputs.append({"target": target})

    np.random.seed(NP_SEED)
    n_samples = len(dict_inputs)
    n_val = int(n_samples * VAL_SPLIT)

    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_inputs = [dict_inputs[i] for i in train_indices]
    val_inputs = [dict_inputs[i] for i in val_indices]

    if not ZERO_SHOT:
        pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=PREDICTION_LEN,
        )

    quantile_predictions = pipeline.predict(val_inputs, prediction_length=PREDICTION_LEN)

    val_inputs_tensor = torch.stack(
        [torch.tensor(item["target"], device=DEVICE) for item in val_inputs]
    )
    quantile_predictions_tensor = torch.stack(
        [torch.tensor(pred, device=DEVICE) for pred in quantile_predictions]
    )

    # quantile pre tensor shape:
    # num_series, num_variates, num_quantiles, prediction_length
    predictions_tensor = quantile_predictions_tensor.squeeze(1).median(dim=1).values

    mse_loss = torch.nn.functional.mse_loss(
        predictions_tensor, val_inputs_tensor[:, -PREDICTION_LEN:]
    ).item()
    mae_loss = torch.nn.functional.l1_loss(
        predictions_tensor, val_inputs_tensor[:, -PREDICTION_LEN:]
    ).item()

    if ZERO_SHOT:
        print("Zero-Shot Evaluation Results:")
        loss_type = "Test"
    else:
        print("Fine-Tuned Evaluation Results:")
        loss_type = "Validation"
    print(f"Dataset KernelSynth :{loss_type} MSE Loss: {mse_loss:.4f}")
    print(f"Dataset KernelSynth :{loss_type} MAE Loss: {mae_loss:.4f}")

    for data_path in args.data_path:
        data_name = data_path.split("/")[-1].split(".")[0]
        print(f"Validating on {data_name}...")

        df = pd.read_csv(data_path).set_index("date").to_numpy()
        df = df.transpose(1, 0)[None, :]  # num_series, num_variates, seq_length

        n_val = int(df.shape[2] * VAL_SPLIT)
        indices = np.random.permutation(df.shape[2])
        val_indices = indices[:n_val]
        val_inputs = df[:, :, val_indices]

        print(val_inputs.shape)

        if not ZERO_SHOT:
            train_indices = indices[n_val:]
            train_data = df[:, :, train_indices]
            pipeline.fit(train_data, context_length=128, prediction_length=PREDICTION_LEN)

        quantile_predictions = pipeline.predict(val_inputs, prediction_length=PREDICTION_LEN)
        val_inputs_tensor = torch.tensor(val_inputs, device=DEVICE)
        quantile_predictions_tensor = torch.stack(
            [torch.tensor(pred, device=DEVICE) for pred in quantile_predictions]
        )
        # quantile pre tensor shape:
        # num_series, num_variates, num_quantiles, prediction_length
        predictions_tensor = quantile_predictions_tensor.median(dim=2).values

        mse_loss = torch.nn.functional.mse_loss(
            predictions_tensor, val_inputs_tensor[:, :, -PREDICTION_LEN:]
        ).item()
        mae_loss = torch.nn.functional.l1_loss(
            predictions_tensor, val_inputs_tensor[:, :, -PREDICTION_LEN:]
        ).item()

        if ZERO_SHOT:
            print("Zero-Shot Evaluation Results:")
            loss_type = "Test"
        else:
            print("Fine-Tuned Evaluation Results:")
            loss_type = "Validation"
        print(f"Dataset {data_name} :{loss_type} MSE Loss: {mse_loss:.4f}")
        print(f"Dataset {data_name} :{loss_type} MAE Loss: {mae_loss:.4f}")
