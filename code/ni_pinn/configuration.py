from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 4
    hidden_dim: int = 128
    hidden_layers: int = 6
    output_dim: int = 7
    activation: str = "swish"
    learn_parameters: bool = True


@dataclass(frozen=True)
class DataConfig:
    cohort_size: int = 363
    train_size: int = 254
    validation_size: int = 36
    test_size: int = 73
    synthetic_size: int = 10000
    synthetic_train_size: int = 8000
    synthetic_validation_size: int = 1000
    synthetic_test_size: int = 1000
    trajectory_hours: int = 2016
    observations: int = 168
    vascular_train: int = 13
    vascular_validation: int = 3
    vascular_test: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1200
    batch_size: int = 256
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-6
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 100
    interior_collocation: int = 10000
    boundary_collocation: int = 2000
    resample_every: int = 50
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class LossConfig:
    pde: float = 1.0
    data: float = 10.0
    neuroimmune: float = 5.0
    boundary: float = 1.0
    coupling: float = 1.0


@dataclass(frozen=True)
class MultiRateConfig:
    fast_steps: int = 100
    medium_steps: int = 42
    slow_steps: int = 12
    fast_seconds: float = 10.0
    medium_hours: float = 4.0
    slow_days: float = 7.0


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 500000
    environments: int = 16
    clip_ratio: float = 0.2
    discount: float = 0.99
    gae_lambda: float = 0.95
    hidden_dim: int = 256
    hidden_layers: int = 3
    episode_seconds: float = 100.0
    timestep_seconds: float = 0.01


@dataclass(frozen=True)
class RewardConfig:
    navigation: float = 1.0
    therapy: float = 0.3
    collision: float = 5.0
    ne_penalty: float = 1.0
    drug_bonus: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    device: str = "cuda"
    precision: str = "float32"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    multirate: MultiRateConfig = field(default_factory=MultiRateConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


def _construct(cls: type[Any], values: dict[str, Any]) -> Any:
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(values.get("seed", 0)),
        device=str(values.get("device", "cuda")),
        precision=str(values.get("precision", "float32")),
        model=_construct(ModelConfig, values.get("model", {})),
        data=_construct(DataConfig, values.get("data", {})),
        training=_construct(TrainingConfig, values.get("training", {})),
        loss=_construct(LossConfig, values.get("loss", {})),
        multirate=_construct(MultiRateConfig, values.get("multirate", {})),
        ppo=_construct(PPOConfig, values.get("ppo", {})),
        reward=_construct(RewardConfig, values.get("reward", {})),
    )
