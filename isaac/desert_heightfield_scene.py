from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from desert_robot.maps.heightfield import generate_desert_heightfield  # noqa: E402
from desert_robot.utils.terrain_mesh import heightfield_to_triangle_mesh  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "isaac"
OUTPUT_PATH = OUTPUT_DIR / "desert_heightfield_scene.usd"


def main() -> None:
    print("Creating Isaac desert heightfield scene...", flush=True)
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    simulation_app.update()

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD stage was not created")
    print("Stage created.", flush=True)

    terrain = generate_desert_heightfield(
        size=(72, 72),
        resolution=0.3,
        seed=42,
        dune_count=8,
        obstacle_count=18,
    )
    print("Procedural desert heightfield generated.", flush=True)

    _create_world(stage)
    print("World root and physics scene created.", flush=True)
    _add_desert_mesh(stage, terrain.elevation, terrain.resolution)
    print("Desert terrain mesh added.", flush=True)
    _add_obstacle_markers(stage, terrain.elevation, terrain.obstacle_mask, terrain.resolution)
    print("Obstacle markers added.", flush=True)
    _add_rover_placeholder(stage, terrain.elevation, terrain.resolution)
    print("Rover placeholder added.", flush=True)
    _add_goal_marker(stage, terrain.elevation, terrain.resolution)
    print("Goal marker added.", flush=True)
    _add_light(stage)
    _add_camera(stage)
    print("Light and camera added.", flush=True)

    for _ in range(30):
        simulation_app.update()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = stage.GetRootLayer().Export(str(OUTPUT_PATH))
    if not saved or not OUTPUT_PATH.exists():
        raise RuntimeError(f"Failed to save Isaac scene to {OUTPUT_PATH}")

    print(f"Saved Isaac desert scene: {OUTPUT_PATH}", flush=True)


def _create_world(stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")


def _add_desert_mesh(stage, elevation, resolution: float) -> None:
    mesh_data = heightfield_to_triangle_mesh(elevation, resolution=resolution)
    mesh = UsdGeom.Mesh.Define(stage, "/World/DesertTerrain")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in mesh_data.vertices]
    )
    mesh.CreateFaceVertexCountsAttr(mesh_data.face_vertex_counts.tolist())
    mesh.CreateFaceVertexIndicesAttr(mesh_data.face_vertex_indices.tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _add_obstacle_markers(stage, elevation, obstacle_mask, resolution: float) -> None:
    ys, xs = obstacle_mask.nonzero()
    if ys.size == 0:
        return

    sample_count = min(80, ys.size)
    sample_indices = np.linspace(0, ys.size - 1, sample_count, dtype=int)

    height, width = elevation.shape
    x_offset = (width - 1) * resolution * 0.5
    y_offset = (height - 1) * resolution * 0.5

    for obstacle_id, index in enumerate(sample_indices):
        row = int(ys[index])
        col = int(xs[index])
        x = col * resolution - x_offset
        y = row * resolution - y_offset
        z = float(elevation[row, col]) + 0.18

        rock = UsdGeom.Cube.Define(stage, f"/World/Obstacles/Rock_{obstacle_id:03d}")
        rock.CreateSizeAttr(1.0)
        rock.AddScaleOp().Set(Gf.Vec3f(0.22, 0.22, 0.18))
        rock.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        UsdPhysics.CollisionAPI.Apply(rock.GetPrim())


def _add_rover_placeholder(stage, elevation, resolution: float) -> None:
    row, col = 10, 10
    x, y, z = _grid_to_world(row, col, elevation, resolution)
    rover = UsdGeom.Cube.Define(stage, "/World/RoverPlaceholder")
    rover.CreateSizeAttr(1.0)
    rover.AddScaleOp().Set(Gf.Vec3f(0.55, 0.35, 0.18))
    rover.AddTranslateOp().Set(Gf.Vec3f(x, y, z + 0.25))
    UsdPhysics.CollisionAPI.Apply(rover.GetPrim())


def _add_goal_marker(stage, elevation, resolution: float) -> None:
    row, col = elevation.shape[0] - 12, elevation.shape[1] - 12
    x, y, z = _grid_to_world(row, col, elevation, resolution)
    goal = UsdGeom.Sphere.Define(stage, "/World/PlantingGoal")
    goal.CreateRadiusAttr(0.35)
    goal.AddTranslateOp().Set(Gf.Vec3f(x, y, z + 0.45))


def _add_light(stage) -> None:
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(900.0)
    sun.CreateAngleAttr(0.8)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 25.0))


def _add_camera(stage) -> None:
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3f(16.0, -20.0, 10.0))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 38.0))
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))


def _grid_to_world(row: int, col: int, elevation, resolution: float) -> tuple[float, float, float]:
    height, width = elevation.shape
    x = col * resolution - (width - 1) * resolution * 0.5
    y = row * resolution - (height - 1) * resolution * 0.5
    z = float(elevation[row, col])
    return x, y, z


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
