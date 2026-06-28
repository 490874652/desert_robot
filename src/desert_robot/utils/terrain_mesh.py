from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriangleMeshData:
    vertices: np.ndarray
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray


def heightfield_to_triangle_mesh(
    elevation: np.ndarray,
    resolution: float,
    center: bool = True,
) -> TriangleMeshData:
    """Convert a 2D heightfield into an indexed triangle mesh."""
    if elevation.ndim != 2:
        raise ValueError("elevation must be a 2D array")
    if min(elevation.shape) < 2:
        raise ValueError("elevation must be at least 2x2")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    height, width = elevation.shape
    yy, xx = np.mgrid[0:height, 0:width]

    x = xx.astype(np.float32) * resolution
    y = yy.astype(np.float32) * resolution
    if center:
        x -= (width - 1) * resolution * 0.5
        y -= (height - 1) * resolution * 0.5

    vertices = np.column_stack(
        [
            x.reshape(-1),
            y.reshape(-1),
            elevation.astype(np.float32).reshape(-1),
        ]
    ).astype(np.float32)

    indices: list[int] = []
    for row in range(height - 1):
        for col in range(width - 1):
            v00 = row * width + col
            v01 = row * width + col + 1
            v10 = (row + 1) * width + col
            v11 = (row + 1) * width + col + 1
            indices.extend([v00, v10, v11, v00, v11, v01])

    face_vertex_indices = np.asarray(indices, dtype=np.int64)
    face_vertex_counts = np.full(face_vertex_indices.size // 3, 3, dtype=np.int64)

    return TriangleMeshData(
        vertices=vertices,
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
    )
