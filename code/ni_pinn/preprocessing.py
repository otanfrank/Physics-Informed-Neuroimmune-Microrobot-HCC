from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CohortSplit:
    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]


@dataclass
class QuantileReference:
    mean_sorted: FloatArray | None = None

    def fit(self, values: FloatArray) -> "QuantileReference":
        self.mean_sorted = np.sort(values, axis=0).mean(axis=1)
        return self

    def transform(self, values: FloatArray) -> FloatArray:
        if self.mean_sorted is None:
            raise RuntimeError("normalizer has not been fitted")
        ranks = np.argsort(np.argsort(values, axis=0), axis=0)
        reference_positions = np.linspace(0, len(self.mean_sorted) - 1, values.shape[0])
        mapped = np.empty_like(values)
        for column in range(values.shape[1]):
            mapped[:, column] = np.interp(
                ranks[:, column],
                reference_positions,
                self.mean_sorted,
            )
        return mapped


def log_transform(values: FloatArray) -> FloatArray:
    if np.any(values < 0):
        raise ValueError("expression matrix contains negative values")
    return np.log2(values + 1.0)


def arcsine_fraction(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.arcsin(np.sqrt(clipped))


def stratified_split(stages: NDArray[np.str_], seed: int) -> CohortSplit:
    indices = np.arange(len(stages), dtype=np.int64)
    train_validation, test = train_test_split(
        indices,
        test_size=73,
        random_state=seed,
        stratify=stages,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=36,
        random_state=seed,
        stratify=stages[train_validation],
    )
    return CohortSplit(train=train, validation=validation, test=test)


def read_expression(path: str | Path) -> tuple[list[str], list[str], FloatArray]:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    genes = frame.index.astype(str).tolist()
    samples = frame.columns.astype(str).tolist()
    return genes, samples, frame.to_numpy(dtype=np.float64).T


def align_samples(
    expression_samples: list[str],
    clinical_samples: list[str],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    expression_lookup = {sample: index for index, sample in enumerate(expression_samples)}
    clinical_lookup = {sample: index for index, sample in enumerate(clinical_samples)}
    shared = sorted(expression_lookup.keys() & clinical_lookup.keys())
    return (
        np.asarray([expression_lookup[sample] for sample in shared], dtype=np.int64),
        np.asarray([clinical_lookup[sample] for sample in shared], dtype=np.int64),
    )


def validate_fractions(values: FloatArray, tolerance: float = 1e-5) -> None:
    if values.ndim != 2 or values.shape[1] != 22:
        raise ValueError("immune fraction matrix must contain 22 columns")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("immune fractions must lie in the unit interval")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=tolerance):
        raise ValueError("immune fractions must sum to one")
