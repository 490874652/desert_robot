from argparse import ArgumentParser
from pathlib import Path
import traceback

import numpy as np
from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "outputs" / "isaac" / "desert_heightfield_scene.usd"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "isaac"
DEFAULT_CAMERA = "/World/Camera"


def parse_args():
    parser = ArgumentParser(description="Capture one RGB/depth frame from an Isaac USD camera.")
    parser.add_argument(
        "--scene",
        default=str(DEFAULT_SCENE),
        help="USD scene path to open.",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="USD camera prim path to render from.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for camera_rgb.png, camera_depth.npy, and camera_depth_mm.png.",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=8,
        help="Replicator subframes to render before reading annotator data.",
    )
    return parser.parse_args()


args = parse_args()
scene_path = Path(args.scene).expanduser().resolve()
output_dir = Path(args.output_dir).expanduser().resolve()

if not scene_path.exists():
    raise FileNotFoundError(
        f"USD scene does not exist: {scene_path}. "
        "Run isaac/desert_heightfield_scene.py first."
    )
if args.width <= 0 or args.height <= 0:
    raise ValueError("width and height must be positive")
if args.warmup_frames <= 0:
    raise ValueError("warmup-frames must be positive")

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> None:
    print(f"Opening USD scene: {scene_path}", flush=True)
    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(str(scene_path))
    if not success:
        raise RuntimeError(f"Failed to open USD scene: {scene_path}")

    for _ in range(10):
        simulation_app.update()

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD stage was not loaded")

    camera_prim = stage.GetPrimAtPath(args.camera)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Camera prim does not exist: {args.camera}")
    if camera_prim.GetTypeName() != "Camera":
        raise RuntimeError(f"Prim is not a USD Camera: {args.camera}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing {args.width}x{args.height} from {args.camera}", flush=True)
    render_product = rep.create.render_product(args.camera, (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annotator.attach(render_product)
    depth_annotator.attach(render_product)

    rep.orchestrator.step(rt_subframes=args.warmup_frames)

    rgb = _extract_array(rgb_annotator.get_data(), "rgb")
    depth = _extract_array(depth_annotator.get_data(), "distance_to_image_plane").astype(np.float32)

    rgb_path = output_dir / "camera_rgb.png"
    depth_npy_path = output_dir / "camera_depth.npy"
    depth_png_path = output_dir / "camera_depth_mm.png"

    _save_rgb(rgb, rgb_path)
    _save_depth(depth, depth_npy_path, depth_png_path)

    rgb_annotator.detach(render_product)
    depth_annotator.detach(render_product)
    render_product.destroy()
    rep.orchestrator.stop()
    for _ in range(3):
        simulation_app.update()

    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size:
        depth_summary = (
            f"min={float(finite_depth.min()):.3f}m "
            f"max={float(finite_depth.max()):.3f}m "
            f"mean={float(finite_depth.mean()):.3f}m"
        )
    else:
        depth_summary = "no finite depth pixels"

    print(f"Saved RGB: {rgb_path}", flush=True)
    print(f"Saved depth array: {depth_npy_path} ({depth_summary})", flush=True)
    print(f"Saved 16-bit depth PNG in millimeters: {depth_png_path}", flush=True)


def _extract_array(data, annotator_name: str) -> np.ndarray:
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    array = np.asarray(data)
    if array.size == 0:
        raise RuntimeError(f"{annotator_name} annotator returned no data")
    return array


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
    rgb8 = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    Image.fromarray(rgb8, mode="RGB").save(path)


def _save_depth(depth: np.ndarray, npy_path: Path, png_path: Path) -> None:
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

    np.save(npy_path, depth)
    depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(png_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
