from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "outputs" / "isaac"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "outputs" / "perception" / "live_dashboard.png"


def parse_args():
    parser = ArgumentParser(description="Live dashboard for RGB, depth, local costmap, and A* path.")
    parser.add_argument(
        "--capture-dir",
        default=str(DEFAULT_CAPTURE_DIR),
        help="Directory containing camera_rgb.png, camera_depth.npy, and camera_params.json.",
    )
    parser.add_argument("--resolution", type=float, default=0.25, help="Local map resolution in meters.")
    parser.add_argument("--max-depth", type=float, default=60.0, help="Maximum accepted depth in meters.")
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=1.0,
        help="Refresh interval for --watch mode.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh the dashboard from capture files.",
    )
    parser.add_argument(
        "--save-snapshot",
        nargs="?",
        const=str(DEFAULT_SNAPSHOT),
        default=None,
        help="Save one dashboard image. Optionally provide a custom path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.max_depth <= 0:
        raise ValueError("max-depth must be positive")
    if args.refresh_seconds <= 0:
        raise ValueError("refresh-seconds must be positive")

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    if args.watch:
        _run_watch(capture_dir, args)
    else:
        snapshot = Path(args.save_snapshot).expanduser().resolve() if args.save_snapshot else None
        _run_once(capture_dir, args, snapshot)


def _run_watch(capture_dir: Path, args) -> None:
    if _is_noninteractive_backend():
        snapshot = DEFAULT_SNAPSHOT
        print(
            f"Matplotlib backend {matplotlib.get_backend()} is non-interactive; "
            f"refreshing snapshot instead: {snapshot}",
            flush=True,
        )
        while True:
            try:
                _run_once(capture_dir, args, snapshot)
                time.sleep(args.refresh_seconds)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"Dashboard refresh failed: {exc}", flush=True)
                time.sleep(args.refresh_seconds)
        return

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    while True:
        try:
            frame = _build_dashboard_frame(capture_dir, args.resolution, args.max_depth)
            _draw_dashboard(fig, axes, frame, path_fraction=_animated_path_fraction())
            plt.pause(args.refresh_seconds)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"Dashboard refresh failed: {exc}", flush=True)
            time.sleep(args.refresh_seconds)
    plt.ioff()
    plt.close(fig)


def _run_once(capture_dir: Path, args, snapshot: Path | None) -> None:
    frame = _build_dashboard_frame(capture_dir, args.resolution, args.max_depth)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _draw_dashboard(fig, axes, frame, path_fraction=1.0)
    if snapshot:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(snapshot, dpi=160)
        print(f"Saved live dashboard snapshot: {snapshot}")
        plt.close(fig)
    else:
        plt.show()


def _build_dashboard_frame(capture_dir: Path, resolution: float, max_depth: float) -> dict:
    from desert_robot.maps.costmap import build_costmap
    from desert_robot.perception.depth_to_heightfield import (
        camera_model_from_params,
        depth_to_local_heightfield,
    )
    from desert_robot.planning.astar import astar

    rgb_path = capture_dir / "camera_rgb.png"
    depth_path = capture_dir / "camera_depth.npy"
    params_path = capture_dir / "camera_params.json"
    for path in (rgb_path, depth_path, params_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing capture file: {path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    camera = camera_model_from_params(params)
    heightfield = depth_to_local_heightfield(
        depth,
        camera=camera,
        resolution=resolution,
        max_depth=max_depth,
    )
    elevation_for_costmap = _fill_unknown_elevation(heightfield.elevation)
    costmap = build_costmap(
        elevation_for_costmap,
        ~heightfield.observed_mask,
        resolution=heightfield.resolution,
        max_slope=1.0,
    )
    planning_mask = _largest_connected_component(costmap.traversable_mask)
    start = _nearest_in_mask(_default_start(planning_mask), planning_mask)
    goal = _nearest_in_mask(_default_goal(planning_mask), planning_mask)
    result = astar(costmap.cost, start, goal)

    return {
        "rgb": rgb,
        "depth": depth,
        "heightfield": heightfield,
        "costmap": costmap,
        "path": result.path,
        "start": start,
        "goal": goal,
        "found": result.found,
        "visited_count": result.visited_count,
        "total_cost": result.total_cost,
        "capture_dir": str(capture_dir),
    }


def _draw_dashboard(fig, axes, frame: dict, path_fraction: float) -> None:
    for ax in axes.ravel():
        ax.clear()

    ax_rgb, ax_depth, ax_heightfield, ax_costmap = axes.ravel()

    ax_rgb.imshow(frame["rgb"])
    ax_rgb.set_title("Isaac RGB")
    ax_rgb.set_axis_off()

    depth = np.ma.masked_invalid(frame["depth"])
    ax_depth.imshow(depth, cmap="magma")
    ax_depth.set_title("Isaac depth (m)")
    ax_depth.set_axis_off()

    heightfield = frame["heightfield"]
    elevation = np.ma.masked_invalid(heightfield.elevation)
    ax_heightfield.imshow(elevation, cmap="terrain", origin="lower")
    ax_heightfield.imshow(
        np.ma.masked_where(heightfield.observed_mask, heightfield.observed_mask),
        cmap="gray",
        alpha=0.35,
        origin="lower",
    )
    ax_heightfield.set_title("Local heightfield")
    ax_heightfield.set_xlabel("grid x")
    ax_heightfield.set_ylabel("grid y")

    costmap = frame["costmap"]
    visible_cost = np.ma.masked_where(~costmap.traversable_mask, costmap.cost)
    ax_costmap.imshow(visible_cost, cmap="viridis", origin="lower")
    ax_costmap.imshow(
        np.ma.masked_where(costmap.traversable_mask, costmap.traversable_mask),
        cmap="gray",
        alpha=0.55,
        origin="lower",
    )
    _draw_path(ax_costmap, frame, path_fraction)
    ax_costmap.set_title("Local costmap + A* path")
    ax_costmap.set_xlabel("grid x")
    ax_costmap.set_ylabel("grid y")

    fig.suptitle(
        f"capture={frame['capture_dir']} | path_cells={len(frame['path'])} | "
        f"visited={frame['visited_count']} | cost={frame['total_cost']:.2f}"
    )
    fig.tight_layout()
    fig.canvas.draw_idle()


def _draw_path(ax, frame: dict, path_fraction: float) -> None:
    path = frame["path"]
    if not path:
        return
    path_array = np.asarray(path)
    count = max(1, int(np.ceil(len(path_array) * np.clip(path_fraction, 0.0, 1.0))))
    visible_path = path_array[:count]
    ax.plot(visible_path[:, 1], visible_path[:, 0], color="white", linewidth=2.0)
    start = frame["start"]
    goal = frame["goal"]
    ax.scatter([start[1]], [start[0]], c="lime", s=45)
    ax.scatter([goal[1]], [goal[0]], c="red", s=45)


def _fill_unknown_elevation(elevation: np.ndarray) -> np.ndarray:
    filled = elevation.copy()
    finite = np.isfinite(filled)
    if not finite.any():
        return np.zeros_like(filled, dtype=np.float32)
    filled[~finite] = float(np.median(filled[finite]))
    return filled.astype(np.float32)


def _default_start(mask: np.ndarray) -> tuple[int, int]:
    height, width = mask.shape
    return height - 1, width // 2


def _default_goal(mask: np.ndarray) -> tuple[int, int]:
    _, width = mask.shape
    return 0, width // 2


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


def _animated_path_fraction() -> float:
    return (time.monotonic() % 4.0) / 4.0


def _is_noninteractive_backend() -> bool:
    return "agg" in matplotlib.get_backend().lower()


if __name__ == "__main__":
    main()
