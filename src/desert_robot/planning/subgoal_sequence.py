from dataclasses import dataclass

import numpy as np

from desert_robot.maps.costmap import CostMap
from desert_robot.planning.astar import AStarResult, astar
from desert_robot.planning.subgoals import GridPoint, SubgoalCandidate, generate_local_subgoals


@dataclass(frozen=True)
class SubgoalSegment:
    subgoal: SubgoalCandidate
    path: list[GridPoint]
    path_cost: float
    visited_count: int


@dataclass(frozen=True)
class SubgoalSequence:
    start: GridPoint
    segments: list[SubgoalSegment]

    @property
    def subgoals(self) -> list[SubgoalCandidate]:
        return [segment.subgoal for segment in self.segments]

    @property
    def path(self) -> list[GridPoint]:
        if not self.segments:
            return [self.start]
        merged: list[GridPoint] = []
        for segment in self.segments:
            if merged:
                merged.extend(segment.path[1:])
            else:
                merged.extend(segment.path)
        return merged

    @property
    def total_cost(self) -> float:
        return float(sum(segment.path_cost for segment in self.segments))


def plan_subgoal_sequence(
    costmap: CostMap,
    start: GridPoint,
    heading: tuple[float, float] = (-1.0, 0.0),
    max_steps: int = 3,
    candidates_per_step: int = 16,
    visited_radius_cells: float = 8.0,
) -> SubgoalSequence:
    """Greedily choose a short reachable sequence of local subgoals."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if candidates_per_step <= 0:
        raise ValueError("candidates_per_step must be positive")
    if visited_radius_cells < 0:
        raise ValueError("visited_radius_cells must be non-negative")
    if not costmap.traversable_mask[start]:
        raise ValueError(f"start point {start} is not traversable")

    current = start
    current_heading = heading
    selected_points = [start]
    segments: list[SubgoalSegment] = []

    for _ in range(max_steps):
        candidates = generate_local_subgoals(
            costmap,
            start=current,
            heading=current_heading,
            max_candidates=candidates_per_step,
        )
        ranked = _rank_candidates(candidates, selected_points, visited_radius_cells)
        segment = _first_reachable_segment(costmap, current, ranked)
        if segment is None:
            break

        segments.append(segment)
        selected_points.append(segment.subgoal.point)
        next_heading = _heading_from_path(segment.path)
        if next_heading is not None:
            current_heading = next_heading
        current = segment.subgoal.point

    return SubgoalSequence(start=start, segments=segments)


def _rank_candidates(
    candidates: list[SubgoalCandidate],
    selected_points: list[GridPoint],
    visited_radius_cells: float,
) -> list[SubgoalCandidate]:
    ranked = []
    for candidate in candidates:
        nearest_selected = min(_distance(candidate.point, point) for point in selected_points)
        if nearest_selected < visited_radius_cells:
            continue
        novelty_bonus = min(nearest_selected / max(visited_radius_cells, 1.0), 2.0) * 0.12
        ranked.append((candidate.score + novelty_bonus, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]


def _first_reachable_segment(
    costmap: CostMap,
    start: GridPoint,
    candidates: list[SubgoalCandidate],
) -> SubgoalSegment | None:
    best: tuple[float, SubgoalCandidate, AStarResult] | None = None
    for candidate in candidates:
        result = astar(costmap.cost, start, candidate.point)
        if not result.found:
            continue
        path_efficiency = candidate.distance_cells / max(len(result.path), 1)
        combined_score = candidate.score + 0.18 * path_efficiency - 0.0008 * result.total_cost
        if best is None or combined_score > best[0]:
            best = (combined_score, candidate, result)
    if best is None:
        return None

    _, candidate, result = best
    return SubgoalSegment(
        subgoal=candidate,
        path=result.path,
        path_cost=float(result.total_cost),
        visited_count=int(result.visited_count),
    )


def _heading_from_path(path: list[GridPoint]) -> tuple[float, float] | None:
    if len(path) < 2:
        return None
    lookback = path[max(0, len(path) - 6)]
    end = path[-1]
    vector = np.asarray([end[0] - lookback[0], end[1] - lookback[1]], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    vector = vector / norm
    return float(vector[0]), float(vector[1])


def _distance(a: GridPoint, b: GridPoint) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
