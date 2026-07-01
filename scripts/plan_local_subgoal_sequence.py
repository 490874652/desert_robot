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
    parser = ArgumentParser(description="Plan a short reachable sequence of local subgoals.")
    parser.add_argument(
        "--costmap",
        default=str(DEFAULT_COSTMAP_PATH),
        help="Path to local_costmap.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for local_subgoal_sequence.npz and local_subgoal_sequence.png.",
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
        "--max-steps",
        type=int,
        default=3,
        help="Maximum number of subgoals in the sequence.",
    )
    return parser.parse_args()


def main() -> None:
    from desert_robot.planning.subgoal_sequence import plan_subgoal_sequence

    args = parse_args()
    costmap_path = Path(args.costmap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not costmap_path.exists():
        raise FileNotFoundError(f"Missing local costmap: {costmap_path}")

    data = np.load(costmap_path)
    costmap = _costmap_from_npz(data)
    start = _parse_point(args.start) if args.start else _nearest_in_mask(_default_start(costmap.traversable_mask), costmap.traversable_mask)
    heading = _parse_heading(args.heading)
    sequence = plan_subgoal_sequence(
        costmap,
        start=start,
        heading=heading,
        max_steps=args.max_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "local_subgoal_sequence.npz"
    png_path = output_dir / "local_subgoal_sequence.png"
    _save_sequence(npz_path, sequence, heading)
    _plot_sequence(costmap, sequence, heading, png_path)

    print(f"Saved local subgoal sequence: {npz_path}")
    print(f"Saved visualization: {png_path}")
    print(f"Start: {sequence.start}")
    print(f"Heading: {heading}")
    print(f"Segments: {len(sequence.segments)}")
    print(f"Total path cells: {len(sequence.path)}")
    print(f"Total path cost: {sequence.total_cost:.2f}")
    for index, segment in enumerate(sequence.segments, start=1):
        print(
            f"{index:02d}. {segment.subgoal.kind} point={segment.subgoal.point} "
            f"score={segment.subgoal.score:.3f} path_cells={len(segment.path)} "
            f"path_cost={segment.path_cost:.2f}"
        )


def _costmap_from_npz(data) -> object:
    from desert_robot.maps.costmap import CostMap

    return CostMap(
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


def _save_sequence(path: Path, sequence, heading: tuple[float, float]) -> None:
    subgoal_points = np.asarray([segment.subgoal.point for segment in sequence.segments], dtype=np.int32)
    subgoal_kinds = np.asarray([segment.subgoal.kind for segment in sequence.segments])
    subgoal_scores = np.asarray([segment.subgoal.score for segment in sequence.segments], dtype=np.float32)
    path_array = np.asarray(sequence.path, dtype=np.int32)
    segment_lengths = np.asarray([len(segment.path) for segment in sequence.segments], dtype=np.int32)
    segment_costs = np.asarray([segment.path_cost for segment in sequence.segments], dtype=np.float32)
    np.savez_compressed(
        path,
        start=np.asarray(sequence.start, dtype=np.int32),
        heading=np.asarray(heading, dtype=np.float32),
        subgoal_points=subgoal_points,
        subgoal_kinds=subgoal_kinds,
        subgoal_scores=subgoal_scores,
        path=path_array,
        segment_lengths=segment_lengths,
        segment_costs=segment_costs,
        total_cost=np.asarray(sequence.total_cost, dtype=np.float32),
    )


def _plot_sequence(costmap, sequence, heading: tuple[float, float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    visible_cost = np.ma.masked_where(~costmap.traversable_mask, costmap.cost)
    axes[0].imshow(visible_cost, cmap="viridis", origin="lower")
    axes[0].imshow(
        np.ma.masked_where(costmap.traversable_mask, costmap.traversable_mask),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    _draw_sequence(axes[0], sequence, heading)
    axes[0].set_title("Subgoal sequence on local cost")
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
    _draw_sequence(axes[1], sequence, heading)
    axes[1].set_title("Subgoal sequence on traversability")
    axes[1].set_xlabel("grid x")
    axes[1].set_ylabel("grid y")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def _draw_sequence(ax, sequence, heading: tuple[float, float]) -> None:
    kind_colors = {
        "safe_frontier": "cyan",
        "climb_entry": "orange",
        "gap": "white",
    }
    kind_markers = {
        "safe_frontier": "o",
        "climb_entry": "^",
        "gap": "s",
    }
    path = np.asarray(sequence.path, dtype=np.int32)
    if path.size:
        ax.plot(path[:, 1], path[:, 0], color="deepskyblue", linewidth=2.2, label="sequence path")
    ax.scatter([sequence.start[1]], [sequence.start[0]], c="lime", s=80, marker="*", label="start")
    ax.arrow(
        sequence.start[1],
        sequence.start[0],
        heading[1] * 8.0,
        heading[0] * 8.0,
        color="lime",
        width=0.12,
        head_width=1.2,
        length_includes_head=True,
    )
    for index, segment in enumerate(sequence.segments, start=1):
        point = segment.subgoal.point
        ax.scatter(
            [point[1]],
            [point[0]],
            c=kind_colors.get(segment.subgoal.kind, "white"),
            s=72,
            marker=kind_markers.get(segment.subgoal.kind, "o"),
            edgecolors="black",
            linewidths=0.7,
        )
        ax.text(
            point[1] + 0.9,
            point[0] + 0.9,
            f"{index}",
            color="black",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.9},
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
