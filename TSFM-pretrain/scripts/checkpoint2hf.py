def convert_checkpoint(checkpoint_path, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    os.makedirs(output_dir, exist_ok=True)
    config_data = checkpoint["config"]

    if hasattr(config_data, "__dict__"):
        config_data = config_data.__dict__

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved config to {config_path}")

    state_dict = checkpoint["state_dict"]

    # OPTIONAL: Handle 'module.' prefix if model was trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

    safetensors_path = os.path.join(output_dir, "model.safetensors")

    save_file(state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    print("Conversion complete!")


if __name__ == "__main__":
    import argparse
    import json
    import os
    import pathlib

    import torch
    import yaml
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file.",
        required=True,
    )
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        help="Path to the YAML configuration file.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory where the converted model will be saved.",
        required=True,
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    if args.yaml:
        with open(args.yaml, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)

        output_dir.mkdir(parents=True, exist_ok=True)

        json_config_path = output_dir / "config.json"
        with open(json_config_path, "w") as json_file:
            json.dump(yaml_config, json_file, indent=4)

        print(f"YAML configuration converted to JSON and saved at: {json_config_path}")

    convert_checkpoint(args.checkpoint, output_dir)
