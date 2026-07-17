from dataclasses import dataclass
from enum import Enum

from torch import Tensor
from torch.optim import Optimizer

from ni_pinn.configuration import LossConfig, MultiRateConfig
from ni_pinn.dynamics import governing_residuals
from ni_pinn.losses import coupling_loss, mean_square, observation_loss
from ni_pinn.networks import NeuroImmunePINN
from ni_pinn.sampling import CollocationSampler


class Scale(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


@dataclass(frozen=True)
class ScaleResult:
    scale: Scale
    loss: float
    iterations: int


@dataclass(frozen=True)
class Window:
    start: float
    end: float


class MultiRateStepper:
    def __init__(
        self,
        model: NeuroImmunePINN,
        optimizer: Optimizer,
        sampler: CollocationSampler,
        schedule: MultiRateConfig,
        weights: LossConfig,
        collocation_count: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.sampler = sampler
        self.schedule = schedule
        self.weights = weights
        self.collocation_count = collocation_count

    def _step(self, loss: Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def fast(self, window: Window, boundary_state: Tensor | None = None) -> ScaleResult:
        total = 0.0
        for index in range(self.schedule.fast_steps):
            fraction_start = index / self.schedule.fast_steps
            fraction_end = (index + 1) / self.schedule.fast_steps
            start = window.start + fraction_start * (window.end - window.start)
            end = window.start + fraction_end * (window.end - window.start)
            points = self.sampler.temporal_window(self.collocation_count, start, end)
            residual = governing_residuals(self.model, points).fast()
            loss = self.weights.pde * mean_square(residual)
            if boundary_state is not None:
                loss = loss + self.weights.coupling * coupling_loss(
                    self.model(points[: len(boundary_state)]),
                    boundary_state,
                )
            total += self._step(loss)
        return ScaleResult(Scale.FAST, total / self.schedule.fast_steps, self.schedule.fast_steps)

    def medium(self, window: Window, fast_boundary: Tensor | None = None) -> ScaleResult:
        total = 0.0
        for index in range(self.schedule.medium_steps):
            fraction_start = index / self.schedule.medium_steps
            fraction_end = (index + 1) / self.schedule.medium_steps
            start = window.start + fraction_start * (window.end - window.start)
            end = window.start + fraction_end * (window.end - window.start)
            points = self.sampler.temporal_window(self.collocation_count, start, end)
            residual = governing_residuals(self.model, points).medium()
            loss = self.weights.pde * mean_square(residual)
            if fast_boundary is not None:
                prediction = self.model(points[: len(fast_boundary)])
                loss = loss + self.weights.coupling * coupling_loss(prediction, fast_boundary)
            total += self._step(loss)
        return ScaleResult(
            Scale.MEDIUM,
            total / self.schedule.medium_steps,
            self.schedule.medium_steps,
        )

    def slow(
        self,
        window: Window,
        observation_coordinates: Tensor,
        observation_targets: Tensor,
        medium_boundary: Tensor | None = None,
    ) -> ScaleResult:
        points = self.sampler.temporal_window(self.collocation_count, window.start, window.end)
        residual = governing_residuals(self.model, points).slow()
        data = observation_loss(self.model(observation_coordinates), observation_targets)
        loss = self.weights.pde * mean_square(residual) + self.weights.data * data
        if medium_boundary is not None:
            prediction = self.model(points[: len(medium_boundary)])
            loss = loss + self.weights.coupling * coupling_loss(prediction, medium_boundary)
        value = self._step(loss)
        return ScaleResult(Scale.SLOW, value, 1)

    def cycle(
        self,
        slow_index: int,
        observation_coordinates: Tensor,
        observation_targets: Tensor,
    ) -> tuple[ScaleResult, ScaleResult, ScaleResult]:
        slow_hours = self.schedule.slow_days * 24.0
        slow_window = Window(slow_index * slow_hours, (slow_index + 1) * slow_hours)
        medium_result = self.medium(slow_window)
        medium_boundary_coordinates = self.sampler.temporal_window(
            min(self.collocation_count, 256),
            slow_window.start,
            slow_window.end,
        )
        medium_boundary = self.model(medium_boundary_coordinates).detach()
        fast_result = self.fast(slow_window, medium_boundary)
        slow_result = self.slow(
            slow_window,
            observation_coordinates,
            observation_targets,
            medium_boundary,
        )
        return fast_result, medium_result, slow_result
