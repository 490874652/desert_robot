import numpy as np

from desert_robot.perception.depth_to_heightfield import (
    CameraModel,
    camera_model_from_params,
    depth_to_local_heightfield,
    depth_to_world_points,
    world_points_to_heightfield,
)


def test_camera_model_from_params_reads_capture_json_shape() -> None:
    params = {
        "resolution": {"width": 4, "height": 3},
        "intrinsics": {"fx": 2.0, "fy": 3.0, "cx": 1.5, "cy": 1.0},
        "world_transform": np.eye(4).tolist(),
    }

    camera = camera_model_from_params(params)

    assert camera.width == 4
    assert camera.height == 3
    assert camera.fx == 2.0
    assert camera.fy == 3.0
    np.testing.assert_allclose(camera.world_transform, np.eye(4))


def test_depth_to_world_points_uses_usd_camera_minus_z_axis() -> None:
    camera = CameraModel(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        width=2,
        height=2,
        world_transform=np.eye(4, dtype=np.float64),
    )
    depth = np.array([[2.0, np.inf], [0.0, 3.0]], dtype=np.float32)

    points = depth_to_world_points(depth, camera=camera)

    np.testing.assert_allclose(points, [[0.0, 0.0, -2.0], [3.0, -3.0, -3.0]])


def test_depth_to_world_points_applies_world_transform() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[3, :3] = [10.0, 20.0, 30.0]
    camera = CameraModel(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        width=1,
        height=1,
        world_transform=transform,
    )
    depth = np.array([[2.0]], dtype=np.float32)

    points = depth_to_world_points(depth, camera=camera)

    np.testing.assert_allclose(points, [[10.0, 20.0, 28.0]])


def test_world_points_to_heightfield_aggregates_points_by_cell() -> None:
    points = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 3.0],
            [1.2, 0.2, 5.0],
        ],
        dtype=np.float32,
    )

    heightfield = world_points_to_heightfield(
        points,
        resolution=1.0,
        bounds_xy=(0.0, 0.0, 2.0, 2.0),
        statistic="median",
    )

    assert heightfield.elevation.shape == (2, 2)
    assert heightfield.observed_mask[0, 0]
    assert heightfield.observed_mask[0, 1]
    assert not heightfield.observed_mask[1, 0]
    np.testing.assert_allclose(heightfield.elevation[0, 0], 2.0)
    np.testing.assert_allclose(heightfield.elevation[0, 1], 5.0)


def test_depth_to_local_heightfield_returns_empty_grid_when_no_valid_depth() -> None:
    camera = CameraModel(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        width=2,
        height=2,
        world_transform=np.eye(4, dtype=np.float64),
    )
    depth = np.full((2, 2), np.inf, dtype=np.float32)

    heightfield = depth_to_local_heightfield(depth, camera=camera, resolution=0.5)

    assert heightfield.elevation.shape == (0, 0)
    assert heightfield.observed_mask.shape == (0, 0)
