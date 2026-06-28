from pathlib import Path
import sys

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "isaac"
OUTPUT_PATH = OUTPUT_DIR / "hello_isaac_scene.usd"


def main() -> None:
    print("Creating a new Isaac stage...", flush=True)
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    simulation_app.update()

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD stage was not created")

    _create_world(stage)
    print("World created. Stepping simulation...", flush=True)

    for _ in range(120):
        simulation_app.update()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = stage.GetRootLayer().Export(str(OUTPUT_PATH))
    if not saved or not OUTPUT_PATH.exists():
        raise RuntimeError(f"Failed to save Isaac scene to {OUTPUT_PATH}")
    print(f"Saved Isaac scene: {OUTPUT_PATH}")
    sys.stdout.flush()


def _create_world(stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    _add_ground(stage)
    _add_cube(stage)
    _add_light(stage)
    _add_camera(stage)


def _add_ground(stage) -> None:
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.05))
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.05))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


def _add_cube(stage) -> None:
    cube = UsdGeom.Cube.Define(stage, "/World/FallingCube")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 2.0))

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(1.0)


def _add_light(stage) -> None:
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(600.0)
    light.CreateAngleAttr(0.7)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))


def _add_camera(stage) -> None:
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3f(5.0, -6.0, 4.0))
    camera.AddOrientOp().Set(Gf.Quatf(0.822, 0.332, 0.443, 0.126))
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
