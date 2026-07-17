from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class Swish(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value * torch.sigmoid(value)


def activation(name: str) -> nn.Module:
    choices: dict[str, type[nn.Module]] = {
        "swish": Swish,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in choices:
        raise ValueError(f"unsupported activation {name}")
    return choices[name]()


class DenseStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(width, hidden_dim), activation(activation_name)))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PositiveParameters(nn.Module):
    names = (
        "tumor_growth",
        "carrying_capacity",
        "cd8_killing",
        "mdsc_support",
        "cd8_source",
        "cd8_proliferation",
        "tumor_half_saturation",
        "cd8_decay",
        "beta2_suppression",
        "macrophage_source",
        "ach_modulation",
        "macrophage_capacity",
        "macrophage_decay",
        "mdsc_source",
        "ne_mdsc_recruitment",
        "mdsc_decay",
        "ne_diffusion",
        "ach_diffusion",
        "ne_decay",
        "ach_decay",
        "drug_diffusion",
        "drug_uptake",
    )

    def __init__(self) -> None:
        super().__init__()
        initial = torch.tensor(
            [
                0.012,
                1.1,
                0.05,
                0.005,
                0.02,
                0.04,
                0.3,
                0.01,
                0.04,
                0.02,
                0.02,
                1.0,
                0.01,
                0.01,
                0.005,
                0.02,
                3e-5,
                2e-5,
                0.04,
                0.5,
                1e-5,
                0.01,
            ],
            dtype=torch.float32,
        )
        self.raw = nn.Parameter(torch.log(torch.expm1(initial)))

    def forward(self) -> dict[str, Tensor]:
        values = torch.nn.functional.softplus(self.raw)
        return dict(zip(self.names, values, strict=True))


class NeuroImmunePINN(nn.Module):
    state_names = ("tumor", "cd8", "macrophage", "mdsc", "ne", "ach", "drug")

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        hidden_layers: int = 6,
        output_dim: int = 7,
        activation_name: str = "swish",
    ) -> None:
        super().__init__()
        self.solution = DenseStack(
            input_dim,
            hidden_dim,
            hidden_layers,
            output_dim,
            activation_name,
        )
        self.parameters_physical = PositiveParameters()
        self.register_buffer("lower", torch.tensor([-1.0, -1.0, -1.0, 0.0]))
        self.register_buffer("upper", torch.tensor([1.0, 1.0, 1.0, 2016.0]))

    def normalize(self, coordinates: Tensor) -> Tensor:
        return 2.0 * (coordinates - self.lower) / (self.upper - self.lower) - 1.0

    def forward(self, coordinates: Tensor) -> Tensor:
        raw = self.solution(self.normalize(coordinates))
        return torch.nn.functional.softplus(raw)

    def named_states(self, coordinates: Tensor) -> dict[str, Tensor]:
        prediction = self(coordinates)
        return {
            name: prediction[..., index : index + 1] for index, name in enumerate(self.state_names)
        }


class TherapyActorCritic(nn.Module):
    def __init__(self, state_dim: int = 13, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.actor = DenseStack(state_dim, hidden_dim, 3, action_dim, "tanh")
        self.critic = DenseStack(state_dim, hidden_dim, 3, 1, "tanh")
        self.log_standard_deviation = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, state: Tensor) -> Normal:
        mean = torch.tanh(self.actor(state))
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        return Normal(mean, standard_deviation)

    def forward(self, state: Tensor) -> tuple[Normal, Tensor]:
        return self.distribution(state), self.critic(state).squeeze(-1)

    def act(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self(state)
        action = distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_probability, value

    def evaluate(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self(state)
        log_probability = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value


class Ensemble(nn.Module):
    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        values = torch.stack([member(inputs) for member in self.members])
        return values.mean(dim=0), values.std(dim=0, unbiased=False)
