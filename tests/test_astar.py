import numpy as np
import pytest

from desert_robot.planning.astar import astar


def test_astar_finds_path_around_wall_gap() -> None:
    costmap = np.ones((12, 12), dtype=np.float32)
    costmap[6, :] = 1_000_000.0
    costmap[6, 5] = 1.0

    result = astar(costmap, (1, 1), (10, 10))

    assert result.found
    assert result.path[0] == (1, 1)
    assert result.path[-1] == (10, 10)
    assert (6, 5) in result.path


def test_astar_returns_empty_path_when_goal_blocked() -> None:
    costmap = np.ones((8, 8), dtype=np.float32)
    costmap[7, 7] = 1_000_000.0

    result = astar(costmap, (0, 0), (7, 7))

    assert not result.found
    assert result.path == []


def test_astar_rejects_out_of_bounds_points() -> None:
    costmap = np.ones((8, 8), dtype=np.float32)

    with pytest.raises(ValueError):
        astar(costmap, (-1, 0), (7, 7))
