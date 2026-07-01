import numpy as np

from desert_robot.maps.heightfield import (
    TERRAIN_BARCHAN_DUNE,
    TERRAIN_FIXED_SHRUB_DUNE,
    TERRAIN_SMALL_VEGETATION,
    generate_desert_heightfield,
)


def test_generate_desert_heightfield_shapes_and_types() -> None:
    terrain = generate_desert_heightfield(size=(32, 48), resolution=0.5, seed=1)

    assert terrain.elevation.shape == (32, 48)
    assert terrain.obstacle_mask.shape == (32, 48)
    assert terrain.soil_looseness.shape == (32, 48)
    assert terrain.bearing_capacity.shape == (32, 48)
    assert terrain.terrain_class.shape == (32, 48)
    assert terrain.fixed_shrub_dune_mask.shape == (32, 48)
    assert terrain.barchan_dune_mask.shape == (32, 48)
    assert terrain.small_vegetation_mask.shape == (32, 48)
    assert terrain.elevation.dtype == np.float32
    assert terrain.obstacle_mask.dtype == bool
    assert terrain.soil_looseness.dtype == np.float32
    assert terrain.bearing_capacity.dtype == np.float32
    assert terrain.terrain_class.dtype == np.uint8
    assert terrain.fixed_shrub_dune_mask.dtype == bool
    assert terrain.barchan_dune_mask.dtype == bool
    assert terrain.small_vegetation_mask.dtype == bool
    assert terrain.resolution == 0.5
    assert np.isfinite(terrain.elevation).all()
    assert np.logical_and(terrain.soil_looseness >= 0.0, terrain.soil_looseness <= 1.0).all()
    assert np.logical_and(terrain.bearing_capacity >= 0.0, terrain.bearing_capacity <= 1.0).all()


def test_generate_desert_heightfield_is_reproducible() -> None:
    first = generate_desert_heightfield(size=(32, 32), seed=11)
    second = generate_desert_heightfield(size=(32, 32), seed=11)

    np.testing.assert_allclose(first.elevation, second.elevation)
    np.testing.assert_array_equal(first.obstacle_mask, second.obstacle_mask)
    np.testing.assert_allclose(first.soil_looseness, second.soil_looseness)
    np.testing.assert_allclose(first.bearing_capacity, second.bearing_capacity)
    np.testing.assert_array_equal(first.terrain_class, second.terrain_class)


def test_generate_desert_heightfield_adds_semantic_terrain_features() -> None:
    terrain = generate_desert_heightfield(
        size=(96, 96),
        seed=7,
        dune_count=0,
        obstacle_count=0,
        fixed_shrub_dune_count=1,
        barchan_dune_count=1,
        small_vegetation_count=4,
    )

    assert terrain.fixed_shrub_dune_mask.any()
    assert terrain.barchan_dune_mask.any()
    assert terrain.small_vegetation_mask.any()
    assert terrain.obstacle_mask[terrain.fixed_shrub_dune_mask].all()
    assert terrain.obstacle_mask[terrain.small_vegetation_mask].all()
    assert np.any(terrain.terrain_class == TERRAIN_FIXED_SHRUB_DUNE)
    assert np.any(terrain.terrain_class == TERRAIN_BARCHAN_DUNE)
    assert np.any(terrain.terrain_class == TERRAIN_SMALL_VEGETATION)
    assert terrain.soil_looseness[terrain.barchan_dune_mask].mean() > terrain.soil_looseness.mean()
