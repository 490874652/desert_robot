from dataclasses import dataclass

import numpy as np

from desert_robot.vehicles import TrackedRoverConfig, default_tracked_rover

TRAVERSABILITY_SAFE = 0
TRAVERSABILITY_CAUTIOUS = 1
TRAVERSABILITY_MARGINAL = 2
TRAVERSABILITY_BLOCKED = 3


@dataclass(frozen=True)
class CostMap:
    cost: np.ndarray
    slope: np.ndarray
    climb_risk: np.ndarray
    side_slope_risk: np.ndarray
    slope_margin: np.ndarray
    traversability_class: np.ndarray
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
    max_slope: float | None = None,
    rover_config: TrackedRoverConfig | None = None,
    obstacle_cost: float = 1_000_000.0,
) -> CostMap:
    """Build a multi-risk desert traversability cost map."""
    if elevation.shape != obstacle_mask.shape:
        raise ValueError("elevation and obstacle_mask must have the same shape")
    if max_slope is not None and max_slope <= 0:
        raise ValueError("max_slope must be positive")

    rover_config = rover_config or default_tracked_rover()
    limit_slope = float(max_slope) if max_slope is not None else rover_config.limit_climb_slope
    safe_slope = min(rover_config.safe_climb_slope, limit_slope)
    conservative_slope = min(rover_config.conservative_climb_slope, safe_slope)

    soil_looseness = _optional_layer(soil_looseness, elevation.shape, default=0.0, name="soil_looseness")
    bearing_capacity = _optional_layer(
        bearing_capacity,
        elevation.shape,
        default=1.0,
        name="bearing_capacity",
    )

    slope = compute_slope(elevation, resolution)
    normalized_slope = np.clip(slope / limit_slope, 0.0, 1.0)
    climb_risk = _normalized_interval(slope, conservative_slope, limit_slope)
    side_slope_risk = np.clip(slope / rover_config.side_slope_limit, 0.0, 1.0)
    slope_margin = (limit_slope - slope).astype(np.float32)
    traversability_class = _classify_traversability(
        slope=slope,
        conservative_slope=conservative_slope,
        safe_slope=safe_slope,
        limit_slope=limit_slope,
    )

    slip_risk = np.clip(0.65 * soil_looseness + 0.35 * normalized_slope, 0.0, 1.0)
    sinkage_risk = np.clip(0.75 * soil_looseness + 0.25 * (1.0 - bearing_capacity), 0.0, 1.0)
    planting_suitability = np.clip(
        1.0 - 0.45 * normalized_slope - 0.35 * sinkage_risk - 0.20 * obstacle_mask.astype(float),
        0.0,
        1.0,
    )

    cost = (
        1.0
        + 5.0 * normalized_slope**2
        + 4.0 * slip_risk
        + 6.0 * sinkage_risk
        + 5.0 * climb_risk**2
        + 2.0 * side_slope_risk**2
    )
    blocked = obstacle_mask | (slope > limit_slope) | (sinkage_risk > 0.88)
    traversability_class[blocked] = TRAVERSABILITY_BLOCKED
    cost = cost.astype(np.float32)
    cost[blocked] = obstacle_cost

    return CostMap(
        cost=cost,
        slope=slope,
        climb_risk=climb_risk.astype(np.float32),
        side_slope_risk=side_slope_risk.astype(np.float32),
        slope_margin=slope_margin,
        traversability_class=traversability_class,
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


def _normalized_interval(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (values > high).astype(np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _classify_traversability(
    slope: np.ndarray,
    conservative_slope: float,
    safe_slope: float,
    limit_slope: float,
) -> np.ndarray:
    traversability = np.full(slope.shape, TRAVERSABILITY_SAFE, dtype=np.uint8)
    traversability[slope > conservative_slope] = TRAVERSABILITY_CAUTIOUS
    traversability[slope > safe_slope] = TRAVERSABILITY_MARGINAL
    traversability[slope > limit_slope] = TRAVERSABILITY_BLOCKED
    return traversability
