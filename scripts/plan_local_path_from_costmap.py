from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_COSTMAP_PATH = PROJECT_ROOT / "outputs" / "perception" / "local_costmap.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "perception"


def parse_args():
    parser = ArgumentParser(description="Plan a local A* path on a perception cost map.")
    parser.add_argument(
        "--costmap",
        default=str(DEFAULT_COSTMAP_PATH),
        help="Path to local_costmap.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for local_path.npz and local_path.png.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start cell as row,col. Defaults near the bottom center.",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Optional goal cell as row,col. Defaults near the top center.",
    )
    return parser.parse_args()


def main() -> None:
    from desert_robot.planning.astar import astar

    args = parse_args()
    costmap_path = Path(args.costmap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not costmap_path.exists():
        raise FileNotFoundError(f"Missing local costmap: {costmap_path}")

    data = np.load(costmap_path)
    cost = data["cost"]
    traversable_mask = data["traversable_mask"]
    origin_xy = tuple(float(value) for value in data["origin_xy"])
    resolution = float(data["resolution"])

    if args.start is None and args.goal is None:
        planning_mask = _largest_connected_component(traversable_mask)
        start = _nearest_in_mask(_default_start(traversable_mask), planning_mask)
        goal = _nearest_in_mask(_default_goal(traversable_mask), planning_mask)
    else:
        start = _parse_point(args.start) if args.start else _default_start(traversable_mask)
        goal = _parse_point(args.goal) if args.goal else _default_goal(traversable_mask)
        start = _nearest_traversable(start, traversable_mask)
        goal = _nearest_traversable(goal, traversable_mask)
        if args.goal is None:
            goal = _nearest_reachable(goal, start, traversable_mask)

    result = astar(cost, start, goal)
    if not result.found:
        raise RuntimeError(f"A* could not find a local path from {start} to {goal}")

    output_dir.mkdir(parents=True, exist_ok=True)
    path_npz = output_dir / "local_path.npz"
    path_png = output_dir / "local_path.png"
    path_array = np.asarray(result.path, dtype=np.int32)
    np.savez_compressed(
        path_npz,
        path=path_array,
        start=np.asarray(start, dtype=np.int32),
        goal=np.asarray(goal, dtype=np.int32),
        total_cost=np.asarray(result.total_cost, dtype=np.float32),
        visited_count=np.asarray(result.visited_count, dtype=np.int32),
        origin_xy=np.asarray(origin_xy, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.float32),
    )
    _plot_path(cost, traversable_mask, result.path, start, goal, path_png)

    print(f"Saved local path: {path_npz}")
    print(f"Saved path visualization: {path_png}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Path cells: {len(result.path)}")
    print(f"Total cost: {result.total_cost:.2f}")
    print(f"Visited cells: {result.visited_count}")


def _parse_point(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("point must be formatted as row,col")
    return int(parts[0]), int(parts[1])


def _default_start(traversable_mask: np.ndarray) -> tuple[int, int]:
    height, width = traversable_mask.shape
    return height - 1, width // 2


def _default_goal(traversable_mask: np.ndarray) -> tuple[int, int]:
    _, width = traversable_mask.shape
    return 0, width // 2


def _nearest_traversable(point: tuple[int, int], traversable_mask: np.ndarray) -> tuple[int, int]:
    return _nearest_in_mask(point, traversable_mask)


def _nearest_in_mask(point: tuple[int, int], mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("mask contains no valid cells")

    target = np.asarray(point, dtype=np.float32)
    candidates = np.column_stack([rows, cols]).astype(np.float32)
    distances = np.linalg.norm(candidates - target, axis=1)
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


def _largest_connected_component(traversable_mask: np.ndarray) -> np.ndarray:
    remaining = traversable_mask.copy()
    largest = np.zeros_like(traversable_mask, dtype=bool)
    while remaining.any():
        row, col = np.argwhere(remaining)[0]
        component = _reachable_mask((int(row), int(col)), remaining)
        remaining[component] = False
        if component.sum() > largest.sum():
            largest = component
    if not largest.any():
        raise ValueError("costmap contains no traversable cells")
    return largest


def _nearest_reachable(
    target: tuple[int, int],
    start: tuple[int, int],
    traversable_mask: np.ndarray,
) -> tuple[int, int]:
    reachable = _reachable_mask(start, traversable_mask)
    rows, cols = np.nonzero(reachable)
    if rows.size == 0:
        raise ValueError(f"start cell is not in a traversable connected component: {start}")

    target_array = np.asarray(target, dtype=np.float32)
    candidates = np.column_stack([rows, cols]).astype(np.float32)
    distances = np.linalg.norm(candidates - target_array, axis=1)
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


def _reachable_mask(start: tuple[int, int], traversable_mask: np.ndarray) -> np.ndarray:
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


def _plot_path(
    cost: np.ndarray,
    traversable_mask: np.ndarray,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
) -> None:
    visible_cost = np.ma.masked_where(~traversable_mask, cost)
    path_array = np.asarray(path)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(visible_cost, cmap="viridis", origin="lower")
    ax.imshow(np.ma.masked_where(traversable_mask, traversable_mask), cmap="gray", alpha=0.55, origin="lower")
    ax.plot(path_array[:, 1], path_array[:, 0], color="white", linewidth=2.0)
    ax.scatter([start[1]], [start[0]], c="lime", s=45, label="start")
    ax.scatter([goal[1]], [goal[0]], c="red", s=45, label="goal")
    ax.set_title("Local A* path on perception cost map")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="cost")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
