import numpy as np

from desert_robot.maps.costmap import (
    TRAVERSABILITY_BLOCKED,
    TRAVERSABILITY_CAUTIOUS,
    TRAVERSABILITY_MARGINAL,
    TRAVERSABILITY_SAFE,
    build_costmap,
    compute_slope,
)
from desert_robot.vehicles import TrackedRoverConfig


def test_compute_slope_on_flat_map_is_zero() -> None:
    elevation = np.zeros((12, 12), dtype=np.float32)
    slope = compute_slope(elevation, resolution=0.25)

    np.testing.assert_allclose(slope, 0.0)


def test_build_costmap_blocks_obstacles() -> None:
    elevation = np.zeros((10, 10), dtype=np.float32)
    obstacles = np.zeros((10, 10), dtype=bool)
    obstacles[5, 5] = True

    result = build_costmap(elevation, obstacles, resolution=0.25)

    assert result.cost[5, 5] >= 1_000_000.0
    assert not result.traversable_mask[5, 5]
    assert result.traversability_class[5, 5] == TRAVERSABILITY_BLOCKED
    assert result.traversable_mask[0, 0]


def test_build_costmap_adds_soft_sand_risk_layers() -> None:
    elevation = np.zeros((10, 10), dtype=np.float32)
    obstacles = np.zeros((10, 10), dtype=bool)
    soil = np.zeros((10, 10), dtype=np.float32)
    bearing = np.ones((10, 10), dtype=np.float32)
    soil[4, 4] = 1.0
    bearing[4, 4] = 0.0

    result = build_costmap(
        elevation,
        obstacles,
        resolution=0.25,
        soil_looseness=soil,
        bearing_capacity=bearing,
    )

    assert result.slip_risk[4, 4] > result.slip_risk[0, 0]
    assert result.sinkage_risk[4, 4] > result.sinkage_risk[0, 0]
    assert result.planting_suitability[4, 4] < result.planting_suitability[0, 0]


def test_build_costmap_adds_tracked_rover_traversability_layers() -> None:
    elevation = np.tile(np.arange(8, dtype=np.float32), (8, 1)) * 0.15
    obstacles = np.zeros_like(elevation, dtype=bool)
    rover = TrackedRoverConfig(
        max_safe_climb_deg=20.0,
        max_limit_climb_deg=30.0,
        slope_safety_margin_deg=5.0,
    )

    result = build_costmap(
        elevation,
        obstacles,
        resolution=1.0,
        rover_config=rover,
    )

    assert result.climb_risk.shape == elevation.shape
    assert result.side_slope_risk.shape == elevation.shape
    assert result.slope_margin.shape == elevation.shape
    assert result.traversability_class.shape == elevation.shape
    assert result.traversability_class.dtype == np.uint8
    assert np.all(result.traversability_class == TRAVERSABILITY_SAFE)
    assert np.all(result.slope_margin > 0.0)


def test_build_costmap_classifies_cautious_marginal_and_blocked_slopes() -> None:
    elevation = np.tile(np.array([0.0, 0.25, 0.6, 1.2], dtype=np.float32), (4, 1))
    obstacles = np.zeros_like(elevation, dtype=bool)
    rover = TrackedRoverConfig(
        max_safe_climb_deg=20.0,
        max_limit_climb_deg=30.0,
        slope_safety_margin_deg=5.0,
    )

    result = build_costmap(
        elevation,
        obstacles,
        resolution=1.0,
        rover_config=rover,
    )

    assert result.traversability_class[1, 0] == TRAVERSABILITY_SAFE
    assert result.traversability_class[1, 1] == TRAVERSABILITY_CAUTIOUS
    assert result.traversability_class[1, 2] == TRAVERSABILITY_MARGINAL
    assert result.traversability_class[1, 3] == TRAVERSABILITY_BLOCKED
    assert not result.traversable_mask[1, 3]
