from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Adam

from ni_pinn.configuration import PPOConfig
from ni_pinn.networks import TherapyActorCritic


@dataclass
class Rollout:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    log_probabilities: Tensor
    values: Tensor
    advantages: Tensor | None = None
    returns: Tensor | None = None

    def compute_advantages(self, final_value: Tensor, discount: float, gae_lambda: float) -> None:
        advantages = torch.zeros_like(self.rewards)
        estimate = torch.zeros_like(final_value)
        next_value = final_value
        for time in reversed(range(self.rewards.shape[0])):
            active = 1.0 - self.dones[time]
            delta = self.rewards[time] + discount * next_value * active - self.values[time]
            estimate = delta + discount * gae_lambda * active * estimate
            advantages[time] = estimate
            next_value = self.values[time]
        self.advantages = advantages
        self.returns = advantages + self.values

    def flatten(self) -> "Rollout":
        if self.advantages is None or self.returns is None:
            raise RuntimeError("advantages must be calculated before flattening")
        return Rollout(
            states=self.states.flatten(0, 1),
            actions=self.actions.flatten(0, 1),
            rewards=self.rewards.flatten(0, 1),
            dones=self.dones.flatten(0, 1),
            log_probabilities=self.log_probabilities.flatten(0, 1),
            values=self.values.flatten(0, 1),
            advantages=self.advantages.flatten(0, 1),
            returns=self.returns.flatten(0, 1),
        )


@dataclass(frozen=True)
class PPOReport:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clipped_fraction: float


class PPOTrainer:
    def __init__(
        self,
        policy: TherapyActorCritic,
        config: PPOConfig,
        learning_rate: float = 3e-4,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        maximum_gradient_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = Adam(policy.parameters(), lr=learning_rate)
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.maximum_gradient_norm = maximum_gradient_norm

    def update(self, rollout: Rollout, epochs: int = 10, batch_size: int = 256) -> PPOReport:
        flat = rollout.flatten()
        if flat.advantages is None or flat.returns is None:
            raise RuntimeError("rollout is incomplete")
        advantages = (flat.advantages - flat.advantages.mean()) / (
            flat.advantages.std(unbiased=False) + 1e-8
        )
        count = flat.states.shape[0]
        accumulated = torch.zeros(5, device=flat.states.device)
        updates = 0
        for _ in range(epochs):
            order = torch.randperm(count, device=flat.states.device)
            for start in range(0, count, batch_size):
                indices = order[start : start + batch_size]
                new_log, entropy, value = self.policy.evaluate(
                    flat.states[indices],
                    flat.actions[indices],
                )
                ratio = torch.exp(new_log - flat.log_probabilities[indices])
                unclipped = ratio * advantages[indices]
                clipped = (
                    ratio.clamp(
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (value - flat.returns[indices]).square().mean()
                entropy_mean = entropy.mean()
                objective = (
                    policy_loss
                    + self.value_weight * value_loss
                    - self.entropy_weight * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.maximum_gradient_norm)
                self.optimizer.step()
                approximate_kl = (flat.log_probabilities[indices] - new_log).mean().abs()
                clipped_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                accumulated += torch.stack(
                    (
                        policy_loss.detach(),
                        value_loss.detach(),
                        entropy_mean.detach(),
                        approximate_kl.detach(),
                        clipped_fraction.detach(),
                    )
                )
                updates += 1
        averaged = accumulated / updates
        return PPOReport(*(float(value) for value in averaged))
