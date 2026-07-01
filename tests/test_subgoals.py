import numpy as np
import pytest

from desert_robot.maps.costmap import CostMap, TRAVERSABILITY_CAUTIOUS, TRAVERSABILITY_SAFE
from desert_robot.planning.subgoals import (
    SUBGOAL_CLIMB_ENTRY,
    SUBGOAL_SAFE_FRONTIER,
    generate_local_subgoals,
)


def test_generate_local_subgoals_returns_reachable_frontier_candidates() -> None:
    costmap = _flat_costmap((24, 24))
    start = (20, 12)

    candidates = generate_local_subgoals(costmap, start, heading=(-1.0, 0.0), max_candidates=6)

    assert candidates
    assert candidates[0].kind == SUBGOAL_SAFE_FRONTIER
    assert all(costmap.traversable_mask[candidate.point] for candidate in candidates)
    assert all(candidate.point[0] < start[0] for candidate in candidates)


def test_generate_local_subgoals_marks_cautious_slope_as_climb_entry() -> None:
    costmap = _flat_costmap((30, 30))
    costmap.traversability_class[10:15, 8:22] = TRAVERSABILITY_CAUTIOUS
    costmap.slope_margin[10:15, 8:22] = 0.12
    start = (24, 15)

    candidates = generate_local_subgoals(costmap, start, heading=(-1.0, 0.0), max_candidates=12)

    assert any(candidate.kind == SUBGOAL_CLIMB_ENTRY for candidate in candidates)


def test_generate_local_subgoals_rejects_blocked_start() -> None:
    costmap = _flat_costmap((16, 16))
    costmap.traversable_mask[8, 8] = False

    with pytest.raises(ValueError):
        generate_local_subgoals(costmap, (8, 8))


def test_generate_local_subgoals_respects_max_candidates() -> None:
    costmap = _flat_costmap((40, 40))

    candidates = generate_local_subgoals(costmap, (34, 20), max_candidates=3)

    assert len(candidates) <= 3


def _flat_costmap(shape: tuple[int, int]) -> CostMap:
    cost = np.ones(shape, dtype=np.float32)
    slope = np.zeros(shape, dtype=np.float32)
    traversable = np.ones(shape, dtype=bool)
    traversability_class = np.full(shape, TRAVERSABILITY_SAFE, dtype=np.uint8)
    return CostMap(
        cost=cost,
        slope=slope,
        climb_risk=np.zeros(shape, dtype=np.float32),
        side_slope_risk=np.zeros(shape, dtype=np.float32),
        slope_margin=np.full(shape, 0.4, dtype=np.float32),
        traversability_class=traversability_class,
        slip_risk=np.zeros(shape, dtype=np.float32),
        sinkage_risk=np.zeros(shape, dtype=np.float32),
        planting_suitability=np.ones(shape, dtype=np.float32),
        traversable_mask=traversable,
    )
