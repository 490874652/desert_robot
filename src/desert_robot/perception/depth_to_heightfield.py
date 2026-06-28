from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    world_transform: np.ndarray


@dataclass(frozen=True)
class LocalHeightfield:
    elevation: np.ndarray
    observed_mask: np.ndarray
    origin_xy: tuple[float, float]
    resolution: float


def camera_model_from_params(params: dict) -> CameraModel:
    """Build a camera model from the JSON written by isaac/capture_camera_frame.py."""
    resolution = params["resolution"]
    intrinsics = params["intrinsics"]
    world_transform = np.asarray(params["world_transform"], dtype=np.float64)
    if world_transform.shape != (4, 4):
        raise ValueError("world_transform must be a 4x4 matrix")

    return CameraModel(
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        width=int(resolution["width"]),
        height=int(resolution["height"]),
        world_transform=world_transform,
    )


def depth_to_world_points(
    depth: np.ndarray,
    camera: CameraModel,
    max_depth: float | None = None,
) -> np.ndarray:
    """Project an Isaac camera depth image into world-space XYZ points.

    Isaac/USD cameras look along local -Z. The generated camera transform uses USD's
    row-vector convention, so local homogeneous points are multiplied on the left.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if depth.shape != (camera.height, camera.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match camera resolution "
            f"{(camera.height, camera.width)}"
        )
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive")

    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - camera.cx) * z / camera.fx
    y = -(rows.astype(np.float64) - camera.cy) * z / camera.fy

    local_points = np.column_stack([x, y, -z, np.ones_like(z)])
    world_points = local_points @ camera.world_transform
    return world_points[:, :3].astype(np.float32)


def world_points_to_heightfield(
    points: np.ndarray,
    resolution: float,
    bounds_xy: tuple[float, float, float, float] | None = None,
    statistic: str = "median",
) -> LocalHeightfield:
    """Rasterize world-space points into a local XY heightfield."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if statistic not in {"median", "mean", "min", "max"}:
        raise ValueError("statistic must be one of: median, mean, min, max")
    if points.size == 0:
        return LocalHeightfield(
            elevation=np.empty((0, 0), dtype=np.float32),
            observed_mask=np.empty((0, 0), dtype=bool),
            origin_xy=(0.0, 0.0),
            resolution=resolution,
        )

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        return LocalHeightfield(
            elevation=np.empty((0, 0), dtype=np.float32),
            observed_mask=np.empty((0, 0), dtype=bool),
            origin_xy=(0.0, 0.0),
            resolution=resolution,
        )

    if bounds_xy is None:
        min_x, min_y = points[:, :2].min(axis=0)
        max_x, max_y = points[:, :2].max(axis=0)
        width = max(1, int(np.floor((max_x - min_x) / resolution)) + 1)
        height = max(1, int(np.floor((max_y - min_y) / resolution)) + 1)
    else:
        min_x, min_y, max_x, max_y = bounds_xy
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("bounds_xy must be (min_x, min_y, max_x, max_y)")
        width = max(1, int(np.ceil((max_x - min_x) / resolution)))
        height = max(1, int(np.ceil((max_y - min_y) / resolution)))

    cols = np.floor((points[:, 0] - min_x) / resolution).astype(np.int64)
    rows = np.floor((points[:, 1] - min_y) / resolution).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    z_values = points[in_bounds, 2]

    elevation = np.full((height, width), np.nan, dtype=np.float32)
    observed_mask = np.zeros((height, width), dtype=bool)
    flat_indices = rows * width + cols

    for flat_index in np.unique(flat_indices):
        values = z_values[flat_indices == flat_index]
        row = int(flat_index // width)
        col = int(flat_index % width)
        elevation[row, col] = _aggregate(values, statistic)
        observed_mask[row, col] = True

    return LocalHeightfield(
        elevation=elevation,
        observed_mask=observed_mask,
        origin_xy=(float(min_x), float(min_y)),
        resolution=resolution,
    )


def depth_to_local_heightfield(
    depth: np.ndarray,
    camera: CameraModel,
    resolution: float,
    max_depth: float | None = None,
    bounds_xy: tuple[float, float, float, float] | None = None,
    statistic: str = "median",
) -> LocalHeightfield:
    points = depth_to_world_points(depth, camera=camera, max_depth=max_depth)
    return world_points_to_heightfield(
        points,
        resolution=resolution,
        bounds_xy=bounds_xy,
        statistic=statistic,
    )


def _aggregate(values: np.ndarray, statistic: str) -> float:
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "min":
        return float(np.min(values))
    return float(np.max(values))
