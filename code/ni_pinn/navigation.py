from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import Env, spaces
from numpy.typing import NDArray

from ni_pinn.configuration import RewardConfig
from ni_pinn.networks import NeuroImmunePINN
from ni_pinn.vascular import VascularGraph, integrate_swarm, swarm_acceleration, wall_force

FloatArray = NDArray[np.float64]


@dataclass
class SwarmState:
    position: FloatArray
    velocity: FloatArray
    elapsed: float
    cumulative_ne: float
    collisions: int


class TherapyNavigationEnv(Env[FloatArray, FloatArray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: VascularGraph,
        world_model: NeuroImmunePINN,
        reward_config: RewardConfig,
        device: torch.device,
        timestep: float = 0.01,
        duration: float = 100.0,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.world_model = world_model
        self.reward_config = reward_config
        self.device = device
        self.timestep = timestep
        self.duration = duration
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float64)
        self.mass = 1e-9
        self.moment = np.asarray([0.0, 0.0, 1e-8], dtype=np.float64)
        self.viscosity = 3.5e-3
        self.radius = 5e-6
        self.state = SwarmState(
            position=graph.nodes[0].copy(),
            velocity=np.zeros(3, dtype=np.float64),
            elapsed=0.0,
            cumulative_ne=0.0,
            collisions=0,
        )

    def _fields(self, position: FloatArray) -> tuple[float, float, float, FloatArray]:
        coordinate = torch.tensor(
            [[position[0], position[1], position[2], self.state.elapsed / 3600.0]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        states = self.world_model.named_states(coordinate)
        ne = states["ne"]
        ne_gradient = torch.autograd.grad(ne.sum(), coordinate, create_graph=False)[0][0, :3]
        return (
            float(states["cd8"].detach()),
            float(ne.detach()),
            float(states["drug"].detach()),
            ne_gradient.detach().cpu().numpy().astype(np.float64),
        )

    def _observation(self) -> FloatArray:
        flow = self.graph.flow_at(self.state.position)
        cd8, ne, drug, ne_gradient = self._fields(self.state.position)
        distance = np.linalg.norm(self.graph.tumor_center - self.state.position)
        return np.concatenate(
            (
                self.state.position,
                flow,
                ne_gradient,
                np.asarray([cd8, drug, distance, ne], dtype=np.float64),
            )
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[FloatArray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.state = SwarmState(
            position=self.graph.nodes[0].copy(),
            velocity=np.zeros(3, dtype=np.float64),
            elapsed=0.0,
            cumulative_ne=0.0,
            collisions=0,
        )
        return self._observation(), {}

    def step(
        self,
        action: FloatArray,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, object]]:
        previous_distance = np.linalg.norm(self.graph.tumor_center - self.state.position)
        gradient = np.diag(np.clip(action, -1.0, 1.0))
        flow = self.graph.flow_at(self.state.position)
        wall = wall_force(self.graph, self.state.position)
        acceleration = swarm_acceleration(
            self.mass,
            self.moment,
            gradient,
            self.viscosity,
            self.radius,
            self.state.velocity,
            flow,
            wall,
        )
        position, velocity = integrate_swarm(
            self.state.position,
            self.state.velocity,
            acceleration,
            self.timestep,
        )
        collision = self.graph.distance_to_vessel(position) > 0.0
        self.state.position = position
        self.state.velocity = velocity
        self.state.elapsed += self.timestep
        self.state.collisions += int(collision)
        cd8, ne, drug, _ = self._fields(position)
        self.state.cumulative_ne += ne * self.timestep
        distance = np.linalg.norm(self.graph.tumor_center - position)
        progress = previous_distance - distance
        therapy = cd8 - self.reward_config.ne_penalty * ne + self.reward_config.drug_bonus * drug
        reward = (
            self.reward_config.navigation * progress
            + self.reward_config.therapy * therapy
            - self.reward_config.collision * float(collision)
        )
        terminated = distance <= self.radius
        truncated = self.state.elapsed >= self.duration
        info: dict[str, object] = {
            "distance": float(distance),
            "cd8": cd8,
            "ne": ne,
            "drug": drug,
            "collision": collision,
            "cumulative_ne": self.state.cumulative_ne,
        }
        return self._observation(), float(reward), terminated, truncated, info
