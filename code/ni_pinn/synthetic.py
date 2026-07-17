from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.stats import qmc

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParameterRanges:
    tumor_growth: tuple[float, float] = (0.005, 0.02)
    carrying_capacity: tuple[float, float] = (0.8, 1.5)
    cd8_killing: tuple[float, float] = (0.01, 0.1)
    beta2_suppression: tuple[float, float] = (0.01, 0.1)
    ach_modulation: tuple[float, float] = (0.005, 0.05)
    ne_diffusion: tuple[float, float] = (1e-6, 1e-4)
    ach_diffusion: tuple[float, float] = (1e-6, 1e-4)
    ne_decay: tuple[float, float] = (0.01, 0.1)
    ach_decay: tuple[float, float] = (0.1, 1.0)
    ne_mdsc_recruitment: tuple[float, float] = (0.001, 0.01)

    def names(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__.keys())

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        pairs = [getattr(self, name) for name in self.names()]
        return (
            np.asarray([pair[0] for pair in pairs], dtype=np.float64),
            np.asarray([pair[1] for pair in pairs], dtype=np.float64),
        )


@dataclass(frozen=True)
class FixedParameters:
    mdsc_support: float = 0.005
    cd8_source: float = 0.02
    cd8_proliferation: float = 0.04
    tumor_half_saturation: float = 0.3
    cd8_decay: float = 0.01
    macrophage_source: float = 0.02
    macrophage_capacity: float = 1.0
    macrophage_decay: float = 0.01
    mdsc_source: float = 0.01
    mdsc_decay: float = 0.02
    ne_source: float = 0.04
    ach_source: float = 0.2


@dataclass(frozen=True)
class SyntheticRecord:
    time: FloatArray
    states: FloatArray
    parameters: FloatArray


def sample_parameters(count: int, seed: int, ranges: ParameterRanges) -> FloatArray:
    sampler = qmc.LatinHypercube(d=len(ranges.names()), seed=seed)
    unit = sampler.random(count)
    lower, upper = ranges.bounds()
    return qmc.scale(unit, lower, upper)


def right_hand_side(
    time: float,
    state: FloatArray,
    variable: FloatArray,
    fixed: FixedParameters,
) -> FloatArray:
    del time
    tumor, cd8, macrophage, mdsc, ne, ach = state
    (
        tumor_growth,
        carrying_capacity,
        cd8_killing,
        beta2_suppression,
        ach_modulation,
        ne_diffusion,
        ach_diffusion,
        ne_decay,
        ach_decay,
        ne_mdsc_recruitment,
    ) = variable
    del ne_diffusion, ach_diffusion
    tumor_rate = (
        tumor_growth * tumor * (1.0 - tumor / carrying_capacity)
        - cd8_killing * cd8 * tumor
        + fixed.mdsc_support * mdsc * tumor
    )
    cd8_rate = (
        fixed.cd8_source
        + fixed.cd8_proliferation * tumor / (tumor + fixed.tumor_half_saturation) * cd8
        - fixed.cd8_decay * cd8
        - beta2_suppression * ne * cd8
    )
    macrophage_rate = (
        fixed.macrophage_source
        + ach_modulation * ach * (fixed.macrophage_capacity - macrophage)
        - fixed.macrophage_decay * macrophage
    )
    mdsc_rate = fixed.mdsc_source + ne_mdsc_recruitment * ne - fixed.mdsc_decay * mdsc
    ne_rate = fixed.ne_source - ne_decay * ne
    ach_rate = fixed.ach_source - ach_decay * ach
    return np.asarray(
        [tumor_rate, cd8_rate, macrophage_rate, mdsc_rate, ne_rate, ach_rate],
        dtype=np.float64,
    )


def generate_record(
    parameters: FloatArray,
    initial: FloatArray,
    observations: int = 168,
    duration_hours: float = 2016.0,
) -> SyntheticRecord:
    evaluation_times = np.linspace(0.0, duration_hours, observations, dtype=np.float64)
    solution = solve_ivp(
        right_hand_side,
        (0.0, duration_hours),
        initial,
        args=(parameters, FixedParameters()),
        method="RK45",
        t_eval=evaluation_times,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return SyntheticRecord(
        time=evaluation_times,
        states=solution.y.T,
        parameters=parameters,
    )


def initial_conditions(count: int, seed: int) -> FloatArray:
    generator = np.random.default_rng(seed)
    lower = np.asarray([0.05, 0.02, 0.1, 0.01, 0.01, 0.01])
    upper = np.asarray([0.25, 0.2, 0.8, 0.2, 0.2, 0.5])
    return generator.uniform(lower, upper, size=(count, 6))


def generate_dataset(path: str | Path, count: int = 10000, seed: int = 0) -> None:
    ranges = ParameterRanges()
    parameters = sample_parameters(count, seed, ranges)
    initials = initial_conditions(count, seed + 1)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.create_dataset("time", data=np.linspace(0.0, 2016.0, 168))
        states = handle.create_dataset(
            "states",
            shape=(count, 168, 6),
            dtype=np.float64,
            chunks=(1, 168, 6),
            compression="gzip",
        )
        handle.create_dataset("parameters", data=parameters, compression="gzip")
        handle.create_dataset("initial_conditions", data=initials, compression="gzip")
        handle.attrs["seed"] = seed
        handle.attrs["rtol"] = 1e-8
        handle.attrs["atol"] = 1e-10
        for index in range(count):
            states[index] = generate_record(parameters[index], initials[index]).states


def split_indices(count: int = 10000, seed: int = 0) -> dict[str, NDArray[np.int64]]:
    generator = np.random.default_rng(seed)
    indices = generator.permutation(count)
    return {
        "train": indices[:8000],
        "validation": indices[8000:9000],
        "test": indices[9000:10000],
    }
