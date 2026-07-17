import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ni_pinn.configuration import ExperimentConfig
from ni_pinn.losses import full_loss
from ni_pinn.metrics import l2_relative_error
from ni_pinn.multirate import MultiRateStepper
from ni_pinn.networks import NeuroImmunePINN
from ni_pinn.sampling import CollocationSampler, Domain

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class TrainingBatch:
    observation_coordinates: Tensor
    observation_targets: Tensor
    boundary_coordinates: Tensor
    boundary_targets: Tensor


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    total_loss: float
    validation_error: float
    learning_rate: float
    physics_loss: float
    observation_loss: float
    neuroimmune_loss: float
    boundary_loss: float


@dataclass
class EarlyStopping:
    patience: int
    best: float = float("inf")
    elapsed: int = 0

    def update(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.elapsed = 0
            return False
        self.elapsed += 1
        return self.elapsed >= self.patience


def atomic_save(payload: dict[str, object], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class PINNTrainer:
    def __init__(
        self,
        model: NeuroImmunePINN,
        config: ExperimentConfig,
        output_directory: str | Path,
    ) -> None:
        self.model = model
        self.config = config
        self.output_directory = Path(output_directory)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.minimum_learning_rate,
        )
        self.sampler = CollocationSampler(Domain(), config.seed, self.device)
        self.stopper = EarlyStopping(config.training.early_stopping_patience)
        self.best_error = float("inf")

    def train_epoch(self, epoch: int, batch: TrainingBatch) -> EpochReport:
        self.model.train()
        collocation = self.sampler.interior(self.config.training.interior_collocation)
        terms = full_loss(
            self.model,
            collocation,
            batch.observation_coordinates,
            batch.observation_targets,
            batch.boundary_coordinates,
            batch.boundary_targets,
        )
        total = terms.weighted(self.config.loss)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self.scheduler.step()
        detached = terms.detached()
        with torch.no_grad():
            prediction = self.model(batch.observation_coordinates)
            validation = float(l2_relative_error(prediction, batch.observation_targets))
        return EpochReport(
            epoch=epoch,
            total_loss=float(total.detach()),
            validation_error=validation,
            learning_rate=self.scheduler.get_last_lr()[0],
            physics_loss=detached["physics"],
            observation_loss=detached["observations"],
            neuroimmune_loss=detached["neuroimmune"],
            boundary_loss=detached["boundary"],
        )

    def save(self, epoch: int, report: EpochReport, name: str) -> None:
        atomic_save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "seed": self.config.seed,
                "config": asdict(self.config),
                "report": asdict(report),
            },
            self.output_directory / name,
        )

    def fit(self, batch: TrainingBatch) -> list[EpochReport]:
        history: list[EpochReport] = []
        for epoch in range(self.config.training.epochs):
            report = self.train_epoch(epoch, batch)
            history.append(report)
            if report.validation_error < self.best_error:
                self.best_error = report.validation_error
                self.save(epoch, report, "best.pt")
            if epoch % 50 == 0:
                LOGGER.info(
                    "epoch=%d loss=%.8f validation=%.4f lr=%.8g",
                    epoch,
                    report.total_loss,
                    report.validation_error,
                    report.learning_rate,
                )
            if self.stopper.update(report.validation_error):
                break
        self.save(history[-1].epoch, history[-1], "final.pt")
        self.write_history(history)
        return history

    def fit_multirate(self, batch: TrainingBatch) -> list[dict[str, float | int | str]]:
        stepper = MultiRateStepper(
            self.model,
            self.optimizer,
            self.sampler,
            self.config.multirate,
            self.config.loss,
            self.config.training.batch_size,
        )
        history: list[dict[str, float | int | str]] = []
        for epoch in range(self.config.training.epochs):
            slow_index = epoch % self.config.multirate.slow_steps
            results = stepper.cycle(
                slow_index,
                batch.observation_coordinates,
                batch.observation_targets,
            )
            for result in results:
                history.append(
                    {
                        "epoch": epoch,
                        "scale": result.scale.value,
                        "loss": result.loss,
                        "iterations": result.iterations,
                    }
                )
            self.scheduler.step()
        target = self.output_directory / "multirate_history.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    def write_history(self, history: list[EpochReport]) -> None:
        target = self.output_directory / "history.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([asdict(report) for report in history], indent=2),
            encoding="utf-8",
        )


def restore(
    trainer: PINNTrainer,
    path: str | Path,
) -> int:
    payload = torch.load(path, map_location=trainer.device)
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.scheduler.load_state_dict(payload["scheduler"])
    set_seed(int(payload["seed"]))
    return int(payload["epoch"]) + 1
