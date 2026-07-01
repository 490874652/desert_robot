from dataclasses import dataclass

import numpy as np

from desert_robot.maps.costmap import (
    CostMap,
    TRAVERSABILITY_CAUTIOUS,
    TRAVERSABILITY_SAFE,
)

GridPoint = tuple[int, int]

SUBGOAL_SAFE_FRONTIER = "safe_frontier"
SUBGOAL_CLIMB_ENTRY = "climb_entry"
SUBGOAL_GAP = "gap"


@dataclass(frozen=True)
class SubgoalCandidate:
    point: GridPoint
    kind: str
    score: float
    distance_cells: float
    mean_cost: float
    slope_margin: float
    clearance_cells: float


def generate_local_subgoals(
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float] = (-1.0, 0.0),
    max_candidates: int = 12,
    min_separation_cells: float = 8.0,
) -> list[SubgoalCandidate]:
    """Generate reachable local subgoals inside the currently visible costmap."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if min_separation_cells < 0:
        raise ValueError("min_separation_cells must be non-negative")
    _validate_point(start, costmap.cost.shape, "start")
    if not costmap.traversable_mask[start]:
        raise ValueError(f"start point {start} is not traversable")

    reachable = _reachable_mask(start, costmap.traversable_mask)
    candidates = [
        *_safe_frontier_candidates(costmap, start, heading, reachable),
        *_climb_entry_candidates(costmap, start, heading, reachable),
        *_gap_candidates(costmap, start, heading, reachable),
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return _non_max_suppression(candidates, max_candidates, min_separation_cells)


def _safe_frontier_candidates(
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float],
    reachable: np.ndarray,
) -> list[SubgoalCandidate]:
    height, width = costmap.cost.shape
    edge_band = max(2, min(height, width) // 12)
    edge_mask = np.zeros_like(reachable, dtype=bool)
    edge_mask[:edge_band, :] = True
    edge_mask[-edge_band:, :] = True
    edge_mask[:, :edge_band] = True
    edge_mask[:, -edge_band:] = True
    forward_region = _forward_mask(costmap.cost.shape, start, heading)
    min_forward_distance = max(6.0, min(height, width) * 0.22)
    far_enough = _forward_distance_mask(costmap.cost.shape, start, heading, min_forward_distance)
    mask = (
        reachable
        & edge_mask
        & forward_region
        & far_enough
        & (costmap.traversability_class <= TRAVERSABILITY_CAUTIOUS)
    )
    return _sample_candidates(
        mask=mask,
        costmap=costmap,
        start=start,
        heading=heading,
        kind=SUBGOAL_SAFE_FRONTIER,
        base_score=1.2,
        stride=max(2, min(height, width) // 24),
    )


def _climb_entry_candidates(
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float],
    reachable: np.ndarray,
) -> list[SubgoalCandidate]:
    climb_mask = reachable & (costmap.traversability_class == TRAVERSABILITY_CAUTIOUS)
    nearby_safe = _dilate(costmap.traversability_class == TRAVERSABILITY_SAFE, radius=2)
    mask = climb_mask & nearby_safe
    return _sample_candidates(
        mask=mask,
        costmap=costmap,
        start=start,
        heading=heading,
        kind=SUBGOAL_CLIMB_ENTRY,
        base_score=0.9,
        stride=max(2, min(costmap.cost.shape) // 28),
    )


def _gap_candidates(
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float],
    reachable: np.ndarray,
) -> list[SubgoalCandidate]:
    clearance = _clearance_cells(costmap.traversable_mask)
    narrow_but_passable = (clearance >= 2.0) & (clearance <= 5.0)
    forward_region = _forward_mask(costmap.cost.shape, start, heading)
    mask = (
        reachable
        & narrow_but_passable
        & forward_region
        & (costmap.traversability_class <= TRAVERSABILITY_CAUTIOUS)
    )
    return _sample_candidates(
        mask=mask,
        costmap=costmap,
        start=start,
        heading=heading,
        kind=SUBGOAL_GAP,
        base_score=0.75,
        stride=max(2, min(costmap.cost.shape) // 30),
        clearance=clearance,
    )


def _sample_candidates(
    mask: np.ndarray,
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float],
    kind: str,
    base_score: float,
    stride: int,
    clearance: np.ndarray | None = None,
) -> list[SubgoalCandidate]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []

    sampled = (rows % stride == 0) & (cols % stride == 0)
    if sampled.any():
        rows = rows[sampled]
        cols = cols[sampled]

    result = []
    for row, col in zip(rows, cols, strict=True):
        point = (int(row), int(col))
        distance = _distance(start, point)
        if distance <= 1.0:
            continue
        mean_cost = _local_mean(costmap.cost, point, radius=2, blocked_value=1_000_000.0)
        slope_margin = float(costmap.slope_margin[point])
        clearance_cells = float(clearance[point]) if clearance is not None else _local_clearance(costmap.traversable_mask, point)
        heading_score = _heading_alignment(start, point, heading)
        score = (
            base_score
            + 0.55 * heading_score
            + 0.35 * np.tanh(slope_margin)
            + 0.18 * np.tanh(clearance_cells / 4.0)
            - 0.018 * distance
            - 0.06 * mean_cost
        )
        result.append(
            SubgoalCandidate(
                point=point,
                kind=kind,
                score=float(score),
                distance_cells=float(distance),
                mean_cost=float(mean_cost),
                slope_margin=slope_margin,
                clearance_cells=clearance_cells,
            )
        )
    return result


def _reachable_mask(start: GridPoint, traversable_mask: np.ndarray) -> np.ndarray:
    height, width = traversable_mask.shape
    reachable = np.zeros_like(traversable_mask, dtype=bool)
    stack = [start]
    while stack:
        row, col = stack.pop()
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        if reachable[row, col] or not traversable_mask[row, col]:
            continue
        reachable[row, col] = True
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                stack.append((row + d_row, col + d_col))
    return reachable


def _clearance_cells(traversable_mask: np.ndarray, max_radius: int = 8) -> np.ndarray:
    clearance = np.zeros(traversable_mask.shape, dtype=np.float32)
    for row, col in np.argwhere(traversable_mask):
        best = 0
        for radius in range(1, max_radius + 1):
            r0 = max(0, row - radius)
            r1 = min(traversable_mask.shape[0], row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(traversable_mask.shape[1], col + radius + 1)
            if not traversable_mask[r0:r1, c0:c1].all():
                break
            best = radius
        clearance[row, col] = best
    return clearance


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows, cols, strict=True):
        r0 = max(0, row - radius)
        r1 = min(mask.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(mask.shape[1], col + radius + 1)
        result[r0:r1, c0:c1] = True
    return result


def _forward_mask(
    shape: tuple[int, int],
    start: GridPoint,
    heading: tuple[float, float],
) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    direction = np.asarray(heading, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray([-1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    vectors = np.stack([rows - start[0], cols - start[1]], axis=-1).astype(np.float32)
    return np.tensordot(vectors, direction, axes=([-1], [0])) > 0.0


def _forward_distance_mask(
    shape: tuple[int, int],
    start: GridPoint,
    heading: tuple[float, float],
    min_distance: float,
) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    direction = np.asarray(heading, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray([-1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    vectors = np.stack([rows - start[0], cols - start[1]], axis=-1).astype(np.float32)
    return np.tensordot(vectors, direction, axes=([-1], [0])) >= min_distance


def _heading_alignment(start: GridPoint, point: GridPoint, heading: tuple[float, float]) -> float:
    vector = np.asarray([point[0] - start[0], point[1] - start[1]], dtype=np.float32)
    vector_norm = float(np.linalg.norm(vector))
    heading_vector = np.asarray(heading, dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading_vector))
    if vector_norm <= 1e-6 or heading_norm <= 1e-6:
        return 0.0
    return float(np.dot(vector / vector_norm, heading_vector / heading_norm))


def _local_mean(
    values: np.ndarray,
    point: GridPoint,
    radius: int,
    blocked_value: float,
) -> float:
    row, col = point
    r0 = max(0, row - radius)
    r1 = min(values.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(values.shape[1], col + radius + 1)
    local = values[r0:r1, c0:c1]
    valid = local[local < blocked_value]
    if valid.size == 0:
        return blocked_value
    return float(valid.mean())


def _local_clearance(traversable_mask: np.ndarray, point: GridPoint, max_radius: int = 8) -> float:
    row, col = point
    for radius in range(1, max_radius + 1):
        r0 = max(0, row - radius)
        r1 = min(traversable_mask.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(traversable_mask.shape[1], col + radius + 1)
        if not traversable_mask[r0:r1, c0:c1].all():
            return float(radius - 1)
    return float(max_radius)


def _non_max_suppression(
    candidates: list[SubgoalCandidate],
    max_candidates: int,
    min_separation_cells: float,
) -> list[SubgoalCandidate]:
    selected: list[SubgoalCandidate] = []
    for candidate in candidates:
        if all(_distance(candidate.point, existing.point) >= min_separation_cells for existing in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _validate_point(point: GridPoint, shape: tuple[int, int], name: str) -> None:
    row, col = point
    if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
        raise ValueError(f"{name} point {point} is outside the costmap")


def _distance(a: GridPoint, b: GridPoint) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
