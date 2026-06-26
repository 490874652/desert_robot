from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostMap:
    cost: np.ndarray
    slope: np.ndarray
    slip_risk: np.ndarray
    sinkage_risk: np.ndarray
    planting_suitability: np.ndarray
    traversable_mask: np.ndarray


def compute_slope(elevation: np.ndarray, resolution: float) -> np.ndarray:
    """Compute slope magnitude from a heightfield."""
    if elevation.ndim != 2:
        raise ValueError("elevation must be a 2D array")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    grad_y, grad_x = np.gradient(elevation.astype(np.float32), resolution)
    return np.hypot(grad_x, grad_y).astype(np.float32)


def build_costmap(
    elevation: np.ndarray,
    obstacle_mask: np.ndarray,
    resolution: float,
    soil_looseness: np.ndarray | None = None,
    bearing_capacity: np.ndarray | None = None,
    max_slope: float = 1.2,
    obstacle_cost: float = 1_000_000.0,
) -> CostMap:
    """Build a multi-risk desert traversability cost map."""
    if elevation.shape != obstacle_mask.shape:
        raise ValueError("elevation and obstacle_mask must have the same shape")
    if max_slope <= 0:
        raise ValueError("max_slope must be positive")

    soil_looseness = _optional_layer(soil_looseness, elevation.shape, default=0.0, name="soil_looseness")
    bearing_capacity = _optional_layer(
        bearing_capacity,
        elevation.shape,
        default=1.0,
        name="bearing_capacity",
    )

    slope = compute_slope(elevation, resolution)
    normalized_slope = np.clip(slope / max_slope, 0.0, 1.0)

    slip_risk = np.clip(0.65 * soil_looseness + 0.35 * normalized_slope, 0.0, 1.0)
    sinkage_risk = np.clip(0.75 * soil_looseness + 0.25 * (1.0 - bearing_capacity), 0.0, 1.0)
    planting_suitability = np.clip(
        1.0 - 0.45 * normalized_slope - 0.35 * sinkage_risk - 0.20 * obstacle_mask.astype(float),
        0.0,
        1.0,
    )

    cost = 1.0 + 5.0 * normalized_slope**2 + 4.0 * slip_risk + 6.0 * sinkage_risk
    blocked = obstacle_mask | (slope > max_slope) | (sinkage_risk > 0.88)
    cost = cost.astype(np.float32)
    cost[blocked] = obstacle_cost

    return CostMap(
        cost=cost,
        slope=slope,
        slip_risk=slip_risk.astype(np.float32),
        sinkage_risk=sinkage_risk.astype(np.float32),
        planting_suitability=planting_suitability.astype(np.float32),
        traversable_mask=~blocked,
    )


def _optional_layer(
    layer: np.ndarray | None,
    shape: tuple[int, int],
    default: float,
    name: str,
) -> np.ndarray:
    if layer is None:
        return np.full(shape, default, dtype=np.float32)
    if layer.shape != shape:
        raise ValueError(f"{name} must have the same shape as elevation")
    return np.clip(layer.astype(np.float32), 0.0, 1.0)
