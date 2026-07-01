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
    parser = ArgumentParser(description="Generate local subgoal candidates from a tracked rover costmap.")
    parser.add_argument(
        "--costmap",
        default=str(DEFAULT_COSTMAP_PATH),
        help="Path to local_costmap.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for local_subgoals.npz and local_subgoals.png.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start cell as row,col. Defaults near the bottom center.",
    )
    parser.add_argument(
        "--heading",
        default="-1,0",
        help="Local heading as drow,dcol. Default points toward the top of the grid.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=12,
        help="Maximum number of subgoal candidates to keep.",
    )
    return parser.parse_args()


def main() -> None:
    from desert_robot.maps.costmap import CostMap
    from desert_robot.planning.subgoals import generate_local_subgoals

    args = parse_args()
    costmap_path = Path(args.costmap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not costmap_path.exists():
        raise FileNotFoundError(f"Missing local costmap: {costmap_path}")

    data = np.load(costmap_path)
    costmap = CostMap(
        cost=data["cost"],
        slope=data["slope"],
        climb_risk=data["climb_risk"],
        side_slope_risk=data["side_slope_risk"],
        slope_margin=data["slope_margin"],
        traversability_class=data["traversability_class"],
        slip_risk=np.zeros_like(data["cost"], dtype=np.float32),
        sinkage_risk=np.zeros_like(data["cost"], dtype=np.float32),
        planting_suitability=np.ones_like(data["cost"], dtype=np.float32),
        traversable_mask=data["traversable_mask"],
    )
    start = _parse_point(args.start) if args.start else _nearest_in_mask(_default_start(costmap.traversable_mask), costmap.traversable_mask)
    heading = _parse_heading(args.heading)
    candidates = generate_local_subgoals(
        costmap,
        start=start,
        heading=heading,
        max_candidates=args.max_candidates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "local_subgoals.npz"
    png_path = output_dir / "local_subgoals.png"
    _save_candidates(npz_path, candidates, start, heading)
    _plot_subgoals(costmap, candidates, start, heading, png_path)

    print(f"Saved local subgoals: {npz_path}")
    print(f"Saved visualization: {png_path}")
    print(f"Start: {start}")
    print(f"Heading: {heading}")
    print(f"Candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"{index:02d}. {candidate.kind} point={candidate.point} "
            f"score={candidate.score:.3f} distance={candidate.distance_cells:.1f} "
            f"margin={candidate.slope_margin:.3f}"
        )


def _save_candidates(path: Path, candidates, start: tuple[int, int], heading: tuple[float, float]) -> None:
    points = np.asarray([candidate.point for candidate in candidates], dtype=np.int32)
    scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
    distances = np.asarray([candidate.distance_cells for candidate in candidates], dtype=np.float32)
    slope_margins = np.asarray([candidate.slope_margin for candidate in candidates], dtype=np.float32)
    clearances = np.asarray([candidate.clearance_cells for candidate in candidates], dtype=np.float32)
    kinds = np.asarray([candidate.kind for candidate in candidates])
    np.savez_compressed(
        path,
        points=points,
        kinds=kinds,
        scores=scores,
        distances=distances,
        slope_margins=slope_margins,
        clearances=clearances,
        start=np.asarray(start, dtype=np.int32),
        heading=np.asarray(heading, dtype=np.float32),
    )


def _plot_subgoals(costmap, candidates, start: tuple[int, int], heading: tuple[float, float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    visible_cost = np.ma.masked_where(~costmap.traversable_mask, costmap.cost)
    axes[0].imshow(visible_cost, cmap="viridis", origin="lower")
    axes[0].imshow(
        np.ma.masked_where(costmap.traversable_mask, costmap.traversable_mask),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    _draw_candidates(axes[0], candidates, start, heading)
    axes[0].set_title("Local cost + subgoals")
    axes[0].set_xlabel("grid x")
    axes[0].set_ylabel("grid y")

    classes = np.ma.masked_where(~costmap.traversable_mask, costmap.traversability_class)
    axes[1].imshow(classes, cmap="viridis", vmin=0, vmax=3, origin="lower")
    axes[1].imshow(
        np.ma.masked_where(costmap.traversable_mask, costmap.traversable_mask),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    _draw_candidates(axes[1], candidates, start, heading)
    axes[1].set_title("Tracked traversability + subgoals")
    axes[1].set_xlabel("grid x")
    axes[1].set_ylabel("grid y")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def _draw_candidates(ax, candidates, start: tuple[int, int], heading: tuple[float, float]) -> None:
    colors = {
        "safe_frontier": "cyan",
        "climb_entry": "orange",
        "gap": "white",
    }
    markers = {
        "safe_frontier": "o",
        "climb_entry": "^",
        "gap": "s",
    }
    ax.scatter([start[1]], [start[0]], c="lime", s=70, marker="*", label="start")
    ax.arrow(
        start[1],
        start[0],
        heading[1] * 8.0,
        heading[0] * 8.0,
        color="lime",
        width=0.12,
        head_width=1.2,
        length_includes_head=True,
    )
    for kind in colors:
        selected = [candidate for candidate in candidates if candidate.kind == kind]
        if not selected:
            continue
        rows = [candidate.point[0] for candidate in selected]
        cols = [candidate.point[1] for candidate in selected]
        ax.scatter(
            cols,
            rows,
            c=colors[kind],
            s=58,
            marker=markers[kind],
            edgecolors="black",
            linewidths=0.6,
            label=kind,
        )
    for index, candidate in enumerate(candidates, start=1):
        ax.text(
            candidate.point[1] + 0.8,
            candidate.point[0] + 0.8,
            str(index),
            color="black",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.8},
        )
    ax.legend(loc="upper right", fontsize=8)


def _parse_point(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("point must be formatted as row,col")
    return int(parts[0]), int(parts[1])


def _parse_heading(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("heading must be formatted as drow,dcol")
    return float(parts[0]), float(parts[1])


def _default_start(traversable_mask: np.ndarray) -> tuple[int, int]:
    height, width = traversable_mask.shape
    return height - 1, width // 2


def _nearest_in_mask(point: tuple[int, int], mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("mask contains no valid cells")

    target = np.asarray(point, dtype=np.float32)
    candidates = np.column_stack([rows, cols]).astype(np.float32)
    distances = np.linalg.norm(candidates - target, axis=1)
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


if __name__ == "__main__":
    main()
