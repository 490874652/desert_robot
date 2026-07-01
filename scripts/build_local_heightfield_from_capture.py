from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "outputs" / "isaac"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "perception"


def parse_args():
    parser = ArgumentParser(description="Build a local heightfield from Isaac camera depth capture.")
    parser.add_argument(
        "--capture-dir",
        default=str(DEFAULT_CAPTURE_DIR),
        help="Directory containing camera_depth.npy and camera_params.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for local_heightfield.npz and local_heightfield.png.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Output heightfield grid resolution in meters.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=60.0,
        help="Maximum accepted depth in meters.",
    )
    return parser.parse_args()


def main() -> None:
    from desert_robot.maps.costmap import build_costmap
    from desert_robot.perception.depth_to_heightfield import (
        camera_model_from_params,
        depth_to_local_heightfield,
    )

    args = parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    depth_path = capture_dir / "camera_depth.npy"
    params_path = capture_dir / "camera_params.json"

    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth capture: {depth_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing camera parameters: {params_path}")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.max_depth <= 0:
        raise ValueError("max-depth must be positive")

    depth = np.load(depth_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    camera = camera_model_from_params(params)
    heightfield = depth_to_local_heightfield(
        depth,
        camera=camera,
        resolution=args.resolution,
        max_depth=args.max_depth,
    )
    elevation_for_costmap = _fill_unknown_elevation(heightfield.elevation)
    unknown_obstacles = ~heightfield.observed_mask
    costmap = build_costmap(
        elevation_for_costmap,
        unknown_obstacles,
        resolution=heightfield.resolution,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "local_heightfield.npz"
    png_path = output_dir / "local_heightfield.png"
    costmap_path = output_dir / "local_costmap.npz"
    costmap_png_path = output_dir / "local_costmap.png"
    np.savez_compressed(
        npz_path,
        elevation=heightfield.elevation,
        observed_mask=heightfield.observed_mask,
        origin_xy=np.asarray(heightfield.origin_xy, dtype=np.float32),
        resolution=np.asarray(heightfield.resolution, dtype=np.float32),
    )
    np.savez_compressed(
        costmap_path,
        cost=costmap.cost,
        slope=costmap.slope,
        climb_risk=costmap.climb_risk,
        side_slope_risk=costmap.side_slope_risk,
        slope_margin=costmap.slope_margin,
        traversability_class=costmap.traversability_class,
        traversable_mask=costmap.traversable_mask,
        origin_xy=np.asarray(heightfield.origin_xy, dtype=np.float32),
        resolution=np.asarray(heightfield.resolution, dtype=np.float32),
    )
    _plot_heightfield(heightfield, png_path)
    _plot_costmap(costmap, costmap_png_path)

    observed = int(heightfield.observed_mask.sum())
    total = int(heightfield.observed_mask.size)
    print(f"Saved local heightfield: {npz_path}")
    print(f"Saved visualization: {png_path}")
    print(f"Saved local costmap: {costmap_path}")
    print(f"Saved costmap visualization: {costmap_png_path}")
    print(f"Grid shape: {heightfield.elevation.shape}")
    print(f"Observed cells: {observed}/{total}")
    print(f"Traversable cells: {int(costmap.traversable_mask.sum())}/{costmap.traversable_mask.size}")
    print(f"Origin XY: {heightfield.origin_xy}")
    print(f"Resolution: {heightfield.resolution:.3f} m/cell")


def _plot_heightfield(heightfield, path: Path) -> None:
    elevation = np.ma.masked_invalid(heightfield.elevation)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(elevation, cmap="terrain", origin="lower")
    ax.imshow(
        np.ma.masked_where(heightfield.observed_mask, heightfield.observed_mask),
        cmap="gray",
        alpha=0.35,
        origin="lower",
    )
    ax.set_title("Local heightfield from Isaac depth")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(image, ax=ax, label="world z (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fill_unknown_elevation(elevation: np.ndarray) -> np.ndarray:
    filled = elevation.copy()
    finite = np.isfinite(filled)
    if not finite.any():
        return np.zeros_like(filled, dtype=np.float32)
    filled[~finite] = float(np.median(filled[finite]))
    return filled.astype(np.float32)


def _plot_costmap(costmap, path: Path) -> None:
    visible_cost = np.ma.masked_where(costmap.cost >= 1_000_000.0, costmap.cost)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].imshow(visible_cost, cmap="viridis", origin="lower")
    axes[0].imshow(
        np.ma.masked_where(costmap.cost < 1_000_000.0, costmap.cost),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    axes[0].set_title("Local cost map from Isaac depth")
    axes[0].set_xlabel("grid x")
    axes[0].set_ylabel("grid y")
    fig.colorbar(image, ax=axes[0], label="cost")

    classes = np.ma.masked_where(~costmap.traversable_mask, costmap.traversability_class)
    class_image = axes[1].imshow(classes, cmap="viridis", vmin=0, vmax=3, origin="lower")
    axes[1].imshow(
        np.ma.masked_where(costmap.traversable_mask, costmap.traversable_mask),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    axes[1].set_title("Tracked rover traversability")
    axes[1].set_xlabel("grid x")
    axes[1].set_ylabel("grid y")
    fig.colorbar(class_image, ax=axes[1], label="0 safe, 1 cautious, 2 marginal, 3 blocked")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
