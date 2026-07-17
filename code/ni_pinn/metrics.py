from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats
from torch import Tensor

FloatArray = NDArray[np.float64]


def l2_relative_error(prediction: Tensor, target: Tensor, epsilon: float = 1e-12) -> Tensor:
    numerator = torch.linalg.vector_norm(prediction - target)
    denominator = torch.linalg.vector_norm(target).clamp_min(epsilon)
    return 100.0 * numerator / denominator


def root_mean_square_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((prediction - target).square()))


def mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(prediction - target))


def parameter_recovery_error(estimate: Tensor, truth: Tensor, epsilon: float = 1e-12) -> Tensor:
    return (
        100.0
        * torch.linalg.vector_norm(estimate - truth)
        / torch.linalg.vector_norm(truth).clamp_min(epsilon)
    )


def pearson_correlation(prediction: Tensor, target: Tensor) -> Tensor:
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    numerator = torch.sum(centered_prediction * centered_target)
    denominator = torch.sqrt(
        torch.sum(centered_prediction.square()) * torch.sum(centered_target.square())
    ).clamp_min(1e-12)
    return numerator / denominator


def navigation_efficiency(reached: Tensor) -> Tensor:
    return 100.0 * reached.float().mean()


def therapeutic_efficacy(
    tumor_before: Tensor,
    tumor_after: Tensor,
    cd8_before: Tensor,
    cd8_after: Tensor,
) -> Tensor:
    tumor_reduction = (tumor_before - tumor_after) / tumor_before.clamp_min(1e-12)
    immune_activation = (cd8_after - cd8_before) / cd8_before.clamp_min(1e-12)
    return 0.5 * (tumor_reduction + immune_activation)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def bootstrap_mean(
    values: FloatArray,
    confidence: float = 0.95,
    replicates: int = 10000,
    seed: int = 0,
) -> Interval:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(replicates, len(values)), replace=True)
    means = samples.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(values.mean()),
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
    )


def paired_t_test(first: FloatArray, second: FloatArray) -> tuple[float, float]:
    result = stats.ttest_rel(first, second)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(first: FloatArray, second: FloatArray) -> tuple[float, float]:
    result = stats.wilcoxon(first, second)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_rank_sum(first: FloatArray, second: FloatArray) -> tuple[float, float]:
    result = stats.ranksums(first, second)
    return float(result.statistic), float(result.pvalue)


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    count: int


def summarize(values: FloatArray) -> MetricSummary:
    return MetricSummary(
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values, ddof=1)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        count=len(values),
    )


def per_state_l2(prediction: Tensor, target: Tensor) -> Tensor:
    numerator = torch.linalg.vector_norm(prediction - target, dim=0)
    denominator = torch.linalg.vector_norm(target, dim=0).clamp_min(1e-12)
    return 100.0 * numerator / denominator


def path_length(positions: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def normalized_path_length(positions: FloatArray) -> float:
    direct = np.linalg.norm(positions[-1] - positions[0])
    return path_length(positions) / max(float(direct), 1e-12)


def collision_rate(collisions: NDArray[np.bool_]) -> float:
    return 100.0 * float(np.mean(collisions))


def degradation_ratio(clean: float, perturbed: float) -> float:
    if clean <= 0.0:
        raise ValueError("clean metric must be positive")
    return perturbed / clean
