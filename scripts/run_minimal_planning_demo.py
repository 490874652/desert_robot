from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from desert_robot.maps.costmap import build_costmap
    from desert_robot.maps.heightfield import generate_desert_heightfield
    from desert_robot.planning.astar import astar

    terrain = generate_desert_heightfield(size=(140, 180), resolution=0.25, seed=7)
    costmap = build_costmap(
        terrain.elevation,
        terrain.obstacle_mask,
        terrain.resolution,
        soil_looseness=terrain.soil_looseness,
        bearing_capacity=terrain.bearing_capacity,
        max_slope=1.0,
    )

    start = (10, 10)
    goal = (126, 164)
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

    visible_cost = np.ma.masked_where(costmap.cost >= 1_000_000.0, costmap.cost)
    axes[3].imshow(visible_cost, cmap="viridis")
    axes[3].imshow(
        np.ma.masked_where(costmap.cost < 1_000_000.0, costmap.cost),
        cmap="gray",
        alpha=0.8,
    )
    axes[3].plot(path_x, path_y, color="white", linewidth=2)
    axes[3].scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=45)
    axes[3].set_title("Final cost map")
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
        "average_slip_risk": float(costmap.slip_risk[path_y, path_x].mean()),
        "average_sinkage_risk": float(costmap.sinkage_risk[path_y, path_x].mean()),
        "goal_planting_suitability": float(costmap.planting_suitability[goal]),
    }


if __name__ == "__main__":
    main()
