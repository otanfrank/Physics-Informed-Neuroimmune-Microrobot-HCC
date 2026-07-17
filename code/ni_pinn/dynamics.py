from dataclasses import dataclass

import torch
from torch import Tensor

from ni_pinn.differential import ResidualBundle, component_derivative, laplacian
from ni_pinn.networks import NeuroImmunePINN


@dataclass(frozen=True)
class SourceFields:
    norepinephrine: Tensor
    acetylcholine: Tensor
    drug_release: Tensor


def constant_sources(coordinates: Tensor) -> SourceFields:
    shape = coordinates[..., :1].shape
    return SourceFields(
        norepinephrine=torch.zeros(shape, device=coordinates.device, dtype=coordinates.dtype),
        acetylcholine=torch.zeros(shape, device=coordinates.device, dtype=coordinates.dtype),
        drug_release=torch.zeros(shape, device=coordinates.device, dtype=coordinates.dtype),
    )


def governing_residuals(
    model: NeuroImmunePINN,
    coordinates: Tensor,
    sources: SourceFields | None = None,
) -> ResidualBundle:
    if not coordinates.requires_grad:
        coordinates = coordinates.requires_grad_(True)
    source = constant_sources(coordinates) if sources is None else sources
    states = model.named_states(coordinates)
    parameters = model.parameters_physical()
    tumor = states["tumor"]
    cd8 = states["cd8"]
    macrophage = states["macrophage"]
    mdsc = states["mdsc"]
    ne = states["ne"]
    ach = states["ach"]
    drug = states["drug"]
    tumor_t = component_derivative(tumor, coordinates, 3)
    cd8_t = component_derivative(cd8, coordinates, 3)
    macrophage_t = component_derivative(macrophage, coordinates, 3)
    mdsc_t = component_derivative(mdsc, coordinates, 3)
    ne_t = component_derivative(ne, coordinates, 3)
    ach_t = component_derivative(ach, coordinates, 3)
    drug_t = component_derivative(drug, coordinates, 3)
    tumor_rhs = (
        parameters["tumor_growth"] * tumor * (1.0 - tumor / parameters["carrying_capacity"])
        - parameters["cd8_killing"] * cd8 * tumor
        + parameters["mdsc_support"] * mdsc * tumor
    )
    cd8_rhs = (
        parameters["cd8_source"]
        + parameters["cd8_proliferation"]
        * tumor
        / (tumor + parameters["tumor_half_saturation"])
        * cd8
        - parameters["cd8_decay"] * cd8
        - parameters["beta2_suppression"] * ne * cd8
    )
    macrophage_rhs = (
        parameters["macrophage_source"]
        + parameters["ach_modulation"] * ach * (parameters["macrophage_capacity"] - macrophage)
        - parameters["macrophage_decay"] * macrophage
    )
    mdsc_rhs = (
        parameters["mdsc_source"]
        + parameters["ne_mdsc_recruitment"] * ne
        - parameters["mdsc_decay"] * mdsc
    )
    ne_rhs = (
        parameters["ne_diffusion"] * laplacian(ne, coordinates)
        + source.norepinephrine
        - parameters["ne_decay"] * ne
    )
    ach_rhs = (
        parameters["ach_diffusion"] * laplacian(ach, coordinates)
        + source.acetylcholine
        - parameters["ach_decay"] * ach
    )
    drug_rhs = (
        parameters["drug_diffusion"] * laplacian(drug, coordinates)
        - parameters["drug_uptake"] * tumor * drug
        + source.drug_release
    )
    return ResidualBundle(
        tumor=tumor_t - tumor_rhs,
        cd8=cd8_t - cd8_rhs,
        macrophage=macrophage_t - macrophage_rhs,
        mdsc=mdsc_t - mdsc_rhs,
        ne=ne_t - ne_rhs,
        ach=ach_t - ach_rhs,
        drug=drug_t - drug_rhs,
    )


def neuroimmune_constraints(model: NeuroImmunePINN, coordinates: Tensor) -> Tensor:
    states = model.named_states(coordinates)
    parameters = model.parameters_physical()
    cd8_t = component_derivative(states["cd8"], coordinates, 3)
    macrophage_t = component_derivative(states["macrophage"], coordinates, 3)
    sympathetic = cd8_t + parameters["beta2_suppression"] * states["ne"] * states["cd8"]
    parasympathetic = macrophage_t - parameters["ach_modulation"] * states["ach"] * (
        parameters["macrophage_capacity"] - states["macrophage"]
    )
    return torch.cat((sympathetic, parasympathetic), dim=-1)
