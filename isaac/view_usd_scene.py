from argparse import ArgumentParser
from pathlib import Path

from isaacsim import SimulationApp


DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "outputs" / "isaac" / "desert_heightfield_scene.usd"


def parse_args():
    parser = ArgumentParser(description="Open a USD scene in Isaac Sim GUI.")
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(DEFAULT_SCENE),
        help="Path to the USD scene to open.",
    )
    return parser.parse_args()


args = parse_args()
scene_path = Path(args.scene).expanduser().resolve()
if not scene_path.exists():
    raise FileNotFoundError(f"USD scene does not exist: {scene_path}")

simulation_app = SimulationApp({"headless": False, "open_usd": str(scene_path)})


def main() -> None:
    print(f"Opened USD scene: {scene_path}", flush=True)
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
