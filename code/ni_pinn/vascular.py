from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class VascularGraph:
    nodes: FloatArray
    edges: IntArray
    radii: FloatArray
    velocities: FloatArray
    tumor_center: FloatArray

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape n by 3")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape m by 2")
        if self.velocities.shape != self.nodes.shape:
            raise ValueError("velocity field must align with nodes")
        if len(self.radii) != len(self.edges):
            raise ValueError("each edge requires a radius")
        if np.any(self.edges < 0) or np.any(self.edges >= len(self.nodes)):
            raise ValueError("edge endpoint is out of range")

    def nearest_node(self, position: FloatArray) -> int:
        return int(np.argmin(np.linalg.norm(self.nodes - position, axis=1)))

    def flow_at(self, position: FloatArray) -> FloatArray:
        nearest = self.nearest_node(position)
        return self.velocities[nearest].copy()

    def distance_to_vessel(self, position: FloatArray) -> float:
        distance = np.inf
        for edge, radius in zip(self.edges, self.radii, strict=True):
            start = self.nodes[edge[0]]
            end = self.nodes[edge[1]]
            segment = end - start
            fraction = np.dot(position - start, segment) / max(np.dot(segment, segment), 1e-12)
            projection = start + np.clip(fraction, 0.0, 1.0) * segment
            distance = min(distance, float(np.linalg.norm(position - projection) - radius))
        return distance


def load_vascular_graph(path: str | Path) -> VascularGraph:
    archive = np.load(path)
    graph = VascularGraph(
        nodes=np.asarray(archive["nodes"], dtype=np.float64),
        edges=np.asarray(archive["edges"], dtype=np.int64),
        radii=np.asarray(archive["radii"], dtype=np.float64),
        velocities=np.asarray(archive["velocities"], dtype=np.float64),
        tumor_center=np.asarray(archive["tumor_center"], dtype=np.float64),
    )
    graph.validate()
    return graph


def wall_force(graph: VascularGraph, position: FloatArray, stiffness: float = 1.0) -> FloatArray:
    node_index = graph.nearest_node(position)
    center = graph.nodes[node_index]
    displacement = position - center
    norm = np.linalg.norm(displacement)
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    incident = np.any(graph.edges == node_index, axis=1)
    radius = float(np.mean(graph.radii[incident])) if np.any(incident) else 1.0
    penetration = max(norm - radius, 0.0)
    return -stiffness * penetration * displacement / norm


def magnetic_force(moment: FloatArray, gradient: FloatArray) -> FloatArray:
    if moment.shape != (3,) or gradient.shape != (3, 3):
        raise ValueError("magnetic moment and gradient dimensions are invalid")
    return gradient.T @ moment


def swarm_acceleration(
    mass: float,
    moment: FloatArray,
    field_gradient: FloatArray,
    viscosity: float,
    radius: float,
    velocity: FloatArray,
    flow: FloatArray,
    wall: FloatArray,
) -> FloatArray:
    drag = -6.0 * np.pi * viscosity * radius * (velocity - flow)
    return (magnetic_force(moment, field_gradient) + drag + wall) / mass


def integrate_swarm(
    position: FloatArray,
    velocity: FloatArray,
    acceleration: FloatArray,
    timestep: float,
) -> tuple[FloatArray, FloatArray]:
    next_velocity = velocity + timestep * acceleration
    next_position = position + timestep * next_velocity
    return next_position, next_velocity
