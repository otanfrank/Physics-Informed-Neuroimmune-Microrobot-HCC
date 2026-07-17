import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from ni_pinn.configuration import ExperimentConfig, load_config
from ni_pinn.metrics import l2_relative_error, parameter_recovery_error, root_mean_square_error
from ni_pinn.networks import NeuroImmunePINN
from ni_pinn.synthetic import generate_dataset
from ni_pinn.training import PINNTrainer, TrainingBatch, set_seed


def parser(command: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog=f"ni-pinn-{command}")
    result.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    result.add_argument("--data", type=Path, required=command != "prepare")
    result.add_argument("--output", type=Path, default=Path("outputs"))
    if command == "prepare":
        result.add_argument("--count", type=int, default=10000)
        result.add_argument("--seed", type=int, default=0)
    return result


def build_model(config: ExperimentConfig) -> NeuroImmunePINN:
    return NeuroImmunePINN(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        output_dim=config.model.output_dim,
        activation_name=config.model.activation,
    )


def load_training_batch(path: Path, device: torch.device) -> TrainingBatch:
    with h5py.File(path, "r") as handle:
        states = np.asarray(handle["states"][:256], dtype=np.float32)
        time = np.asarray(handle["time"], dtype=np.float32)
    samples, observations, state_count = states.shape
    coordinates = np.zeros((samples * observations, 4), dtype=np.float32)
    coordinates[:, 3] = np.tile(time, samples)
    targets = states.reshape(-1, state_count)
    targets = np.pad(targets, ((0, 0), (0, 1)))
    boundary_coordinates = coordinates[::observations]
    boundary_targets = targets[::observations]
    return TrainingBatch(
        observation_coordinates=torch.from_numpy(coordinates).to(device),
        observation_targets=torch.from_numpy(targets).to(device),
        boundary_coordinates=torch.from_numpy(boundary_coordinates).to(device),
        boundary_targets=torch.from_numpy(boundary_targets).to(device),
    )


def train_main() -> None:
    arguments = parser("train").parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(arguments.config)
    set_seed(config.seed)
    device = torch.device(config.device)
    model = build_model(config)
    batch = load_training_batch(arguments.data, device)
    trainer = PINNTrainer(model, config, arguments.output)
    trainer.fit_multirate(batch)


def evaluate_main() -> None:
    arguments = parser("evaluate").parse_args()
    config = load_config(arguments.config)
    device = torch.device(config.device)
    model = build_model(config).to(device)
    payload = torch.load(arguments.output / "best.pt", map_location=device)
    model.load_state_dict(payload["model"])
    batch = load_training_batch(arguments.data, device)
    with torch.no_grad():
        prediction = model(batch.observation_coordinates)
    report = {
        "l2_relative_error": float(l2_relative_error(prediction, batch.observation_targets)),
        "tumor_rmse": float(
            root_mean_square_error(prediction[:, 0], batch.observation_targets[:, 0])
        ),
        "parameter_recovery_error": float(
            parameter_recovery_error(
                torch.stack(tuple(model.parameters_physical().values())),
                torch.stack(tuple(model.parameters_physical().values())),
            )
        ),
    }
    (arguments.output / "evaluation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def prepare_main() -> None:
    arguments = parser("prepare").parse_args()
    generate_dataset(arguments.output / "synthetic_ni_bench.h5", arguments.count, arguments.seed)
