from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from desert_robot.maps.costmap import TRAVERSABILITY_MARGINAL  # noqa: E402


def main() -> None:
    from desert_robot.maps.costmap import build_costmap
    from desert_robot.maps.heightfield import generate_desert_heightfield
    from desert_robot.planning.astar import astar
    from desert_robot.vehicles import default_tracked_rover

    terrain = generate_desert_heightfield(size=(140, 180), resolution=0.25, seed=7)
    rover = default_tracked_rover()
    costmap = build_costmap(
        terrain.elevation,
        terrain.obstacle_mask,
        terrain.resolution,
        soil_looseness=terrain.soil_looseness,
        bearing_capacity=terrain.bearing_capacity,
        rover_config=rover,
    )

    planning_mask = _largest_connected_component(costmap.traversable_mask)
    start = _nearest_in_mask((10, 10), planning_mask)
    goal = _nearest_in_mask((126, 164), planning_mask)
    result = astar(costmap.cost, start, goal)
    if not result.found:
        raise RuntimeError("A* failed to find a path. Try a different seed, start, or goal.")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "minimal_planning_demo.png"

    metrics = summarize_path(result.path, terrain.resolution, costmap)
    plot_demo(terrain, costmap, result.path, start, goal)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Path found with {len(result.path)} cells")
    print(f"Path length: {metrics['path_length_m']:.2f} m")
    print(f"Total cost: {result.total_cost:.2f}")
    print(f"Average path cost: {metrics['average_cost']:.2f}")
    print(f"Max slope on path: {metrics['max_slope']:.3f}")
    print(f"Min slope margin on path: {metrics['min_slope_margin']:.3f}")
    print(f"Marginal-or-worse path cells: {metrics['marginal_or_worse_cells']:.0f}")
    print(f"Average slip risk: {metrics['average_slip_risk']:.3f}")
    print(f"Average sinkage risk: {metrics['average_sinkage_risk']:.3f}")
    print(f"Goal planting suitability: {metrics['goal_planting_suitability']:.3f}")
    print(f"Visited cells: {result.visited_count}")
    print(f"Saved visualization: {output_path}")


def plot_demo(
    terrain,
    costmap,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    path_y = [p[0] for p in path]
    path_x = [p[1] for p in path]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    axes[0].imshow(terrain.elevation, cmap="terrain")
    axes[0].imshow(
        np.ma.masked_where(~terrain.obstacle_mask, terrain.obstacle_mask),
        cmap="gray",
        alpha=0.7,
    )
    axes[0].plot(path_x, path_y, color="magenta", linewidth=2)
    axes[0].scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=45)
    axes[0].set_title("Heightfield and A* path")
    axes[0].set_axis_off()

    axes[1].imshow(terrain.soil_looseness, cmap="YlOrBr", vmin=0.0, vmax=1.0)
    axes[1].plot(path_x, path_y, color="cyan", linewidth=2)
    axes[1].set_title("Soil looseness")
    axes[1].set_axis_off()

    axes[2].imshow(costmap.sinkage_risk, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].plot(path_x, path_y, color="white", linewidth=2)
    axes[2].set_title("Sinkage risk")
    axes[2].set_axis_off()

    visible_class = np.ma.masked_where(~costmap.traversable_mask, costmap.traversability_class)
    axes[3].imshow(visible_class, cmap="viridis", vmin=0, vmax=3)
    axes[3].imshow(
        np.ma.masked_where(costmap.cost < 1_000_000.0, costmap.cost),
        cmap="gray",
        alpha=0.8,
    )
    axes[3].plot(path_x, path_y, color="white", linewidth=2)
    axes[3].scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=45)
    axes[3].set_title("Tracked rover traversability")
    axes[3].set_axis_off()


def summarize_path(path: list[tuple[int, int]], resolution: float, costmap) -> dict[str, float]:
    path_array = np.array(path)
    path_y = path_array[:, 0]
    path_x = path_array[:, 1]
    step_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1) * resolution
    goal = path[-1]

    return {
        "path_length_m": float(step_lengths.sum()),
        "average_cost": float(costmap.cost[path_y, path_x].mean()),
        "max_slope": float(costmap.slope[path_y, path_x].max()),
        "min_slope_margin": float(costmap.slope_margin[path_y, path_x].min()),
        "marginal_or_worse_cells": float(
            np.count_nonzero(costmap.traversability_class[path_y, path_x] >= TRAVERSABILITY_MARGINAL)
        ),
        "average_slip_risk": float(costmap.slip_risk[path_y, path_x].mean()),
        "average_sinkage_risk": float(costmap.sinkage_risk[path_y, path_x].mean()),
        "goal_planting_suitability": float(costmap.planting_suitability[goal]),
    }


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


if __name__ == "__main__":
    main()
