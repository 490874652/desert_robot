from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, sqrt

import numpy as np

GridPoint = tuple[int, int]


@dataclass(frozen=True)
class AStarResult:
    path: list[GridPoint]
    total_cost: float
    visited_count: int

    @property
    def found(self) -> bool:
        return bool(self.path)


_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, sqrt(2.0)),
    (-1, 1, sqrt(2.0)),
    (1, -1, sqrt(2.0)),
    (1, 1, sqrt(2.0)),
)


def astar(
    costmap: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
    obstacle_cost: float = 1_000_000.0,
) -> AStarResult:
    """Plan a path on a 2D cost grid using 8-connected A*."""
    if costmap.ndim != 2:
        raise ValueError("costmap must be a 2D array")

    height, width = costmap.shape
    _validate_point(start, height, width, "start")
    _validate_point(goal, height, width, "goal")

    if costmap[start] >= obstacle_cost or costmap[goal] >= obstacle_cost:
        return AStarResult(path=[], total_cost=inf, visited_count=0)

    frontier: list[tuple[float, GridPoint]] = []
    heappush(frontier, (0.0, start))

    came_from: dict[GridPoint, GridPoint | None] = {start: None}
    g_score: dict[GridPoint, float] = {start: 0.0}
    visited_count = 0

    while frontier:
        _, current = heappop(frontier)
        visited_count += 1

        if current == goal:
            return AStarResult(
                path=_reconstruct_path(came_from, goal),
                total_cost=g_score[goal],
                visited_count=visited_count,
            )

        for dy, dx, distance in _NEIGHBORS:
            neighbor = (current[0] + dy, current[1] + dx)
            if not _in_bounds(neighbor, height, width):
                continue
            if costmap[neighbor] >= obstacle_cost:
                continue

            step_cost = float(costmap[neighbor]) * distance
            tentative_g = g_score[current] + step_cost
            if tentative_g >= g_score.get(neighbor, inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            priority = tentative_g + _heuristic(neighbor, goal)
            heappush(frontier, (priority, neighbor))

    return AStarResult(path=[], total_cost=inf, visited_count=visited_count)


def _validate_point(point: GridPoint, height: int, width: int, name: str) -> None:
    if not _in_bounds(point, height, width):
        raise ValueError(f"{name} point {point} is outside the costmap")


def _in_bounds(point: GridPoint, height: int, width: int) -> bool:
    y, x = point
    return 0 <= y < height and 0 <= x < width


def _heuristic(a: GridPoint, b: GridPoint) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _reconstruct_path(
    came_from: dict[GridPoint, GridPoint | None],
    goal: GridPoint,
) -> list[GridPoint]:
    current: GridPoint | None = goal
    path: list[GridPoint] = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path
