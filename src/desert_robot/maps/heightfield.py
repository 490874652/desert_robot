from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Heightfield:
    elevation: np.ndarray
    obstacle_mask: np.ndarray
    soil_looseness: np.ndarray
    bearing_capacity: np.ndarray
    resolution: float


def generate_desert_heightfield(
    size: tuple[int, int] = (128, 128),
    resolution: float = 0.25,
    seed: int | None = None,
    dune_count: int = 7,
    obstacle_count: int = 24,
) -> Heightfield:
    """Generate a simple 2.5D desert terrain with dunes, soil layers, and obstacles."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("size must contain positive dimensions")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    rng = np.random.default_rng(seed)
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]

    elevation = np.zeros(size, dtype=np.float32)

    # Broad sinusoidal structure gives the map a desert-like rolling base.
    elevation += 0.35 * np.sin(xx / width * 2.5 * np.pi + 0.4)
    elevation += 0.22 * np.cos(yy / height * 2.0 * np.pi - 0.7)

    for _ in range(dune_count):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        sx = rng.uniform(width * 0.08, width * 0.22)
        sy = rng.uniform(height * 0.04, height * 0.14)
        amplitude = rng.uniform(0.4, 1.5)
        dune = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
        elevation += amplitude * dune

    roughness = rng.normal(loc=0.0, scale=0.035, size=size).astype(np.float32)
    elevation = (elevation + roughness).astype(np.float32)
    elevation -= float(elevation.min())

    obstacle_mask = np.zeros(size, dtype=bool)
    for _ in range(obstacle_count):
        radius = int(rng.integers(2, 6))
        cx = int(rng.integers(radius, max(radius + 1, width - radius)))
        cy = int(rng.integers(radius, max(radius + 1, height - radius)))
        obstacle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        obstacle_mask |= obstacle

    soil_looseness = _generate_soil_looseness(size=size, rng=rng, xx=xx, yy=yy)
    bearing_capacity = np.clip(1.0 - soil_looseness + rng.normal(0.0, 0.04, size), 0.0, 1.0)

    return Heightfield(
        elevation=elevation,
        obstacle_mask=obstacle_mask,
        soil_looseness=soil_looseness.astype(np.float32),
        bearing_capacity=bearing_capacity.astype(np.float32),
        resolution=resolution,
    )


def _generate_soil_looseness(
    size: tuple[int, int],
    rng: np.random.Generator,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    height, width = size
    soil = np.full(size, 0.25, dtype=np.float32)

    for _ in range(6):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        sx = rng.uniform(width * 0.06, width * 0.18)
        sy = rng.uniform(height * 0.06, height * 0.18)
        softness = rng.uniform(0.25, 0.75)
        patch = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
        soil += softness * patch

    fine_noise = rng.normal(loc=0.0, scale=0.04, size=size)
    return np.clip(soil + fine_noise, 0.0, 1.0)
