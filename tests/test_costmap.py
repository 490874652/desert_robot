import numpy as np

from desert_robot.maps.costmap import build_costmap, compute_slope


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
