import numpy as np
import pytest

from desert_robot.utils.terrain_mesh import heightfield_to_triangle_mesh


def test_heightfield_to_triangle_mesh_counts() -> None:
    elevation = np.zeros((3, 4), dtype=np.float32)

    mesh = heightfield_to_triangle_mesh(elevation, resolution=0.5)

    assert mesh.vertices.shape == (12, 3)
    assert mesh.face_vertex_counts.shape == (12,)
    assert mesh.face_vertex_indices.shape == (36,)
    assert np.all(mesh.face_vertex_counts == 3)


def test_heightfield_to_triangle_mesh_preserves_elevation() -> None:
    elevation = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    mesh = heightfield_to_triangle_mesh(elevation, resolution=1.0, center=False)

    np.testing.assert_allclose(mesh.vertices[:, 2], [0.0, 1.0, 2.0, 3.0])


def test_heightfield_to_triangle_mesh_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        heightfield_to_triangle_mesh(np.zeros((1, 4), dtype=np.float32), resolution=1.0)

    with pytest.raises(ValueError):
        heightfield_to_triangle_mesh(np.zeros((3, 4), dtype=np.float32), resolution=0.0)
