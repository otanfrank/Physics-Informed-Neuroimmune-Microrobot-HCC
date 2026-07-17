from dataclasses import dataclass

import torch
from torch import Tensor


def derivative(value: Tensor, coordinate: Tensor, order: int = 1) -> Tensor:
    result = value
    for _ in range(order):
        gradient = torch.autograd.grad(
            result,
            coordinate,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        result = gradient
    return result


def component_derivative(value: Tensor, coordinates: Tensor, component: int) -> Tensor:
    gradient = derivative(value, coordinates)
    return gradient[..., component : component + 1]


def laplacian(value: Tensor, coordinates: Tensor, dimensions: int = 3) -> Tensor:
    result = torch.zeros_like(value)
    first = derivative(value, coordinates)
    for dimension in range(dimensions):
        component = first[..., dimension : dimension + 1]
        second = derivative(component, coordinates)[..., dimension : dimension + 1]
        result = result + second
    return result


@dataclass(frozen=True)
class ResidualBundle:
    tumor: Tensor
    cd8: Tensor
    macrophage: Tensor
    mdsc: Tensor
    ne: Tensor
    ach: Tensor
    drug: Tensor

    def stack(self) -> Tensor:
        return torch.cat(
            (self.tumor, self.cd8, self.macrophage, self.mdsc, self.ne, self.ach, self.drug),
            dim=-1,
        )

    def fast(self) -> Tensor:
        return torch.cat((self.ne, self.ach), dim=-1)

    def medium(self) -> Tensor:
        return torch.cat((self.cd8, self.macrophage, self.mdsc), dim=-1)

    def slow(self) -> Tensor:
        return torch.cat((self.tumor, self.drug), dim=-1)
