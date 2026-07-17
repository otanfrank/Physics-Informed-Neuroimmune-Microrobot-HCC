from dataclasses import dataclass

import torch
from torch import Tensor

from ni_pinn.configuration import LossConfig
from ni_pinn.dynamics import governing_residuals, neuroimmune_constraints
from ni_pinn.networks import NeuroImmunePINN


@dataclass(frozen=True)
class LossTerms:
    physics: Tensor
    observations: Tensor
    neuroimmune: Tensor
    boundary: Tensor
    coupling: Tensor

    def weighted(self, weights: LossConfig) -> Tensor:
        return (
            weights.pde * self.physics
            + weights.data * self.observations
            + weights.neuroimmune * self.neuroimmune
            + weights.boundary * self.boundary
            + weights.coupling * self.coupling
        )

    def detached(self) -> dict[str, float]:
        return {
            "physics": float(self.physics.detach()),
            "observations": float(self.observations.detach()),
            "neuroimmune": float(self.neuroimmune.detach()),
            "boundary": float(self.boundary.detach()),
            "coupling": float(self.coupling.detach()),
        }


def mean_square(value: Tensor) -> Tensor:
    return value.square().mean()


def relative_square(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    scale = target.square().mean(dim=0, keepdim=True).clamp_min(epsilon)
    return ((prediction - target).square() / scale).mean()


def observation_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    difference = prediction - target
    if mask is not None:
        difference = difference * mask
        denominator = mask.sum().clamp_min(1.0)
        return difference.square().sum() / denominator
    return difference.square().mean()


def boundary_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return mean_square(prediction - target)


def coupling_loss(left: Tensor, right: Tensor) -> Tensor:
    return mean_square(left - right)


def full_loss(
    model: NeuroImmunePINN,
    collocation: Tensor,
    observation_coordinates: Tensor,
    observation_targets: Tensor,
    boundary_coordinates: Tensor,
    boundary_targets: Tensor,
    coupling_left: Tensor | None = None,
    coupling_right: Tensor | None = None,
) -> LossTerms:
    residual = governing_residuals(model, collocation).stack()
    predictions = model(observation_coordinates)
    boundaries = model(boundary_coordinates)
    constraints = neuroimmune_constraints(model, collocation)
    zero = torch.zeros((), device=collocation.device, dtype=collocation.dtype)
    coupling = zero
    if coupling_left is not None and coupling_right is not None:
        coupling = coupling_loss(coupling_left, coupling_right)
    return LossTerms(
        physics=mean_square(residual),
        observations=observation_loss(predictions, observation_targets),
        neuroimmune=mean_square(constraints),
        boundary=boundary_loss(boundaries, boundary_targets),
        coupling=coupling,
    )


def ablated_loss(terms: LossTerms, variant: str) -> LossTerms:
    zero = torch.zeros_like(terms.physics)
    choices = {
        "full": terms,
        "without_neuroimmune": LossTerms(
            terms.physics,
            terms.observations,
            zero,
            terms.boundary,
            terms.coupling,
        ),
        "without_multirate": LossTerms(
            terms.physics,
            terms.observations,
            terms.neuroimmune,
            terms.boundary,
            zero,
        ),
        "data_only": LossTerms(zero, terms.observations, zero, zero, zero),
    }
    if variant not in choices:
        raise ValueError(f"unknown loss variant {variant}")
    return choices[variant]
