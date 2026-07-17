from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Domain:
    spatial_low: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    spatial_high: tuple[float, float, float] = (1.0, 1.0, 1.0)
    time_low: float = 0.0
    time_high: float = 2016.0


class CollocationSampler:
    def __init__(self, domain: Domain, seed: int, device: torch.device) -> None:
        self.domain = domain
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def interior(self, count: int) -> Tensor:
        unit = torch.rand((count, 4), generator=self.generator, device=self.device)
        low = torch.tensor((*self.domain.spatial_low, self.domain.time_low), device=self.device)
        high = torch.tensor((*self.domain.spatial_high, self.domain.time_high), device=self.device)
        return (low + unit * (high - low)).requires_grad_(True)

    def initial(self, count: int) -> Tensor:
        points = self.interior(count).detach()
        points[:, 3] = self.domain.time_low
        return points.requires_grad_(True)

    def boundary(self, count: int) -> Tensor:
        points = self.interior(count).detach()
        dimensions = torch.randint(
            0,
            3,
            (count,),
            generator=self.generator,
            device=self.device,
        )
        sides = torch.randint(
            0,
            2,
            (count,),
            generator=self.generator,
            device=self.device,
        )
        for index in range(count):
            dimension = int(dimensions[index])
            points[index, dimension] = (
                self.domain.spatial_low[dimension]
                if int(sides[index]) == 0
                else self.domain.spatial_high[dimension]
            )
        return points.requires_grad_(True)

    def temporal_window(self, count: int, start: float, end: float) -> Tensor:
        points = self.interior(count).detach()
        points[:, 3] = start + torch.rand(
            count,
            generator=self.generator,
            device=self.device,
        ) * (end - start)
        return points.requires_grad_(True)


def latin_hypercube(count: int, dimensions: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    result = torch.empty(count, dimensions)
    for dimension in range(dimensions):
        permutation = torch.randperm(count, generator=generator)
        jitter = torch.rand(count, generator=generator)
        result[:, dimension] = (permutation + jitter) / count
    return result
