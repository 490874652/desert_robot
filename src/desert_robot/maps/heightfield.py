from dataclasses import dataclass

import numpy as np

TERRAIN_SAND = 0
TERRAIN_FIXED_SHRUB_DUNE = 1
TERRAIN_BARCHAN_DUNE = 2
TERRAIN_SMALL_VEGETATION = 3


@dataclass(frozen=True)
class Heightfield:
    elevation: np.ndarray
    obstacle_mask: np.ndarray
    soil_looseness: np.ndarray
    bearing_capacity: np.ndarray
    terrain_class: np.ndarray
    fixed_shrub_dune_mask: np.ndarray
    barchan_dune_mask: np.ndarray
    small_vegetation_mask: np.ndarray
    resolution: float


def generate_desert_heightfield(
    size: tuple[int, int] = (128, 128),
    resolution: float = 0.25,
    seed: int | None = None,
    dune_count: int = 7,
    obstacle_count: int = 24,
    fixed_shrub_dune_count: int = 2,
    barchan_dune_count: int = 2,
    small_vegetation_count: int = 18,
) -> Heightfield:
    """Generate a 2.5D desert terrain with explicit dune and vegetation semantics."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("size must contain positive dimensions")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    rng = np.random.default_rng(seed)
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]

    elevation = np.zeros(size, dtype=np.float32)
    fixed_shrub_dune_mask = np.zeros(size, dtype=bool)
    barchan_dune_mask = np.zeros(size, dtype=bool)
    small_vegetation_mask = np.zeros(size, dtype=bool)

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

    for _ in range(fixed_shrub_dune_count):
        dune, mask = _fixed_shrub_dune(
            xx=xx,
            yy=yy,
            width=width,
            height=height,
            rng=rng,
        )
        elevation += dune
        fixed_shrub_dune_mask |= mask

    for _ in range(barchan_dune_count):
        dune, mask = _barchan_dune(
            xx=xx,
            yy=yy,
            width=width,
            height=height,
            rng=rng,
        )
        elevation += dune
        barchan_dune_mask |= mask

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

    for _ in range(small_vegetation_count):
        bump, mask = _small_vegetation_bump(
            xx=xx,
            yy=yy,
            width=width,
            height=height,
            rng=rng,
        )
        elevation += bump
        small_vegetation_mask |= mask

    obstacle_mask |= fixed_shrub_dune_mask | small_vegetation_mask

    soil_looseness = _generate_soil_looseness(size=size, rng=rng, xx=xx, yy=yy)
    soil_looseness = np.clip(soil_looseness + 0.25 * barchan_dune_mask, 0.0, 1.0)
    bearing_capacity = np.clip(1.0 - soil_looseness + rng.normal(0.0, 0.04, size), 0.0, 1.0)
    terrain_class = np.full(size, TERRAIN_SAND, dtype=np.uint8)
    terrain_class[barchan_dune_mask] = TERRAIN_BARCHAN_DUNE
    terrain_class[fixed_shrub_dune_mask] = TERRAIN_FIXED_SHRUB_DUNE
    terrain_class[small_vegetation_mask] = TERRAIN_SMALL_VEGETATION

    return Heightfield(
        elevation=elevation,
        obstacle_mask=obstacle_mask,
        soil_looseness=soil_looseness.astype(np.float32),
        bearing_capacity=bearing_capacity.astype(np.float32),
        terrain_class=terrain_class,
        fixed_shrub_dune_mask=fixed_shrub_dune_mask,
        barchan_dune_mask=barchan_dune_mask,
        small_vegetation_mask=small_vegetation_mask,
        resolution=resolution,
    )


def _fixed_shrub_dune(
    xx: np.ndarray,
    yy: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cx = rng.uniform(width * 0.18, width * 0.82)
    cy = rng.uniform(height * 0.18, height * 0.82)
    sx = rng.uniform(width * 0.09, width * 0.16)
    sy = rng.uniform(height * 0.07, height * 0.13)
    amplitude = rng.uniform(1.0, 1.9)

    mound = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    shrub_ring = np.exp(-(((np.hypot((xx - cx) / sx, (yy - cy) / sy) - 0.72) ** 2) / 0.08))
    dune = amplitude * mound + 0.18 * shrub_ring
    mask = mound > 0.22
    return dune.astype(np.float32), mask


def _barchan_dune(
    xx: np.ndarray,
    yy: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cx = rng.uniform(width * 0.2, width * 0.8)
    cy = rng.uniform(height * 0.2, height * 0.8)
    sx = rng.uniform(width * 0.09, width * 0.16)
    sy = rng.uniform(height * 0.06, height * 0.12)
    amplitude = rng.uniform(0.8, 1.6)
    yaw = rng.uniform(-0.7, 0.7)

    dx = xx - cx
    dy = yy - cy
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    along = cos_yaw * dx + sin_yaw * dy
    across = -sin_yaw * dx + cos_yaw * dy

    body = np.exp(-((along**2) / (2 * sx**2) + (across**2) / (2 * sy**2)))
    horn_offset = sx * 0.7
    horn_spread = sy * 0.58
    left_horn = np.exp(-(((along - horn_offset) ** 2) / (2 * (sx * 0.75) ** 2) + ((across - sy) ** 2) / (2 * horn_spread**2)))
    right_horn = np.exp(-(((along - horn_offset) ** 2) / (2 * (sx * 0.75) ** 2) + ((across + sy) ** 2) / (2 * horn_spread**2)))
    slip_face = 1.0 / (1.0 + np.exp(-along / max(sx * 0.12, 1e-6)))

    dune = amplitude * (0.9 * body + 0.42 * left_horn + 0.42 * right_horn)
    dune *= 0.72 + 0.38 * slip_face
    mask = (body + 0.65 * left_horn + 0.65 * right_horn) > 0.22
    return dune.astype(np.float32), mask


def _small_vegetation_bump(
    xx: np.ndarray,
    yy: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    radius = rng.uniform(1.8, 4.2)
    cx = rng.uniform(radius, width - radius)
    cy = rng.uniform(radius, height - radius)
    normalized_radius = np.hypot(xx - cx, yy - cy) / radius
    mask = normalized_radius <= 1.0
    height_scale = rng.uniform(0.18, 0.42)
    bump = np.zeros_like(xx, dtype=np.float32)
    bump[mask] = height_scale * np.sqrt(np.maximum(0.0, 1.0 - normalized_radius[mask] ** 2))
    return bump, mask


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
