import numpy as np

from desert_robot.maps.heightfield import generate_desert_heightfield


def test_generate_desert_heightfield_shapes_and_types() -> None:
    terrain = generate_desert_heightfield(size=(32, 48), resolution=0.5, seed=1)

    assert terrain.elevation.shape == (32, 48)
    assert terrain.obstacle_mask.shape == (32, 48)
    assert terrain.soil_looseness.shape == (32, 48)
    assert terrain.bearing_capacity.shape == (32, 48)
    assert terrain.elevation.dtype == np.float32
    assert terrain.obstacle_mask.dtype == bool
    assert terrain.soil_looseness.dtype == np.float32
    assert terrain.bearing_capacity.dtype == np.float32
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
