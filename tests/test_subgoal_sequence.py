import numpy as np
import pytest

from desert_robot.maps.costmap import CostMap, TRAVERSABILITY_SAFE
from desert_robot.planning.subgoal_sequence import plan_subgoal_sequence


def test_plan_subgoal_sequence_returns_reachable_segments() -> None:
    costmap = _flat_costmap((48, 48))
    start = (42, 24)

    sequence = plan_subgoal_sequence(
        costmap,
        start=start,
        heading=(-1.0, 0.0),
        max_steps=2,
        candidates_per_step=10,
    )

    assert sequence.segments
    assert sequence.path[0] == start
    assert sequence.total_cost > 0.0
    expected_start = start
    for segment in sequence.segments:
        assert segment.path[0] == expected_start
        assert segment.path[-1] == segment.subgoal.point
        assert costmap.traversable_mask[segment.subgoal.point]
        expected_start = segment.subgoal.point


def test_plan_subgoal_sequence_merges_segment_paths() -> None:
    costmap = _flat_costmap((48, 48))
    sequence = plan_subgoal_sequence(costmap, start=(42, 24), max_steps=2)

    expected_length = sum(len(segment.path) for segment in sequence.segments)
    if len(sequence.segments) > 1:
        expected_length -= len(sequence.segments) - 1

    assert len(sequence.path) == expected_length


def test_plan_subgoal_sequence_rejects_blocked_start() -> None:
    costmap = _flat_costmap((20, 20))
    costmap.traversable_mask[12, 10] = False
    costmap.cost[12, 10] = 1_000_000.0

    with pytest.raises(ValueError):
        plan_subgoal_sequence(costmap, start=(12, 10))


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
