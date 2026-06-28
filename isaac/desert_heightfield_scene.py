from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

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
        size=(144, 144),
        resolution=0.15,
        seed=42,
        dune_count=10,
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
    _create_material(stage, "/World/Materials/Sand", (0.74, 0.56, 0.33), roughness=0.92)
    _create_material(stage, "/World/Materials/Rock", (0.28, 0.24, 0.19), roughness=0.86)
    _create_material(stage, "/World/Materials/Rover", (0.12, 0.18, 0.16), roughness=0.65)
    _create_material(stage, "/World/Materials/Goal", (0.18, 0.56, 0.22), roughness=0.55)


def _add_desert_mesh(stage, elevation, resolution: float) -> None:
    mesh_data = heightfield_to_triangle_mesh(elevation, resolution=resolution)
    mesh = UsdGeom.Mesh.Define(stage, "/World/DesertTerrain")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in mesh_data.vertices]
    )
    mesh.CreateFaceVertexCountsAttr(mesh_data.face_vertex_counts.tolist())
    mesh.CreateFaceVertexIndicesAttr(mesh_data.face_vertex_indices.tolist())
    mesh.CreateNormalsAttr(_compute_heightfield_normals(elevation, resolution))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _set_vertex_colors(mesh, elevation)
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

        rock = UsdGeom.Sphere.Define(stage, f"/World/Obstacles/Rock_{obstacle_id:03d}")
        rock.CreateRadiusAttr(0.45)
        scale = 0.28 + 0.08 * ((obstacle_id * 17) % 5) / 4.0
        rock.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        rock.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        UsdPhysics.CollisionAPI.Apply(rock.GetPrim())
        _bind_material(stage, rock.GetPrim(), "/World/Materials/Rock")


def _add_rover_placeholder(stage, elevation, resolution: float) -> None:
    row, col = 10, 10
    x, y, z = _grid_to_world(row, col, elevation, resolution)
    rover = UsdGeom.Cube.Define(stage, "/World/RoverPlaceholder")
    rover.CreateSizeAttr(1.0)
    rover.AddScaleOp().Set(Gf.Vec3f(0.55, 0.35, 0.18))
    rover.AddTranslateOp().Set(Gf.Vec3f(x, y, z + 0.25))
    UsdPhysics.CollisionAPI.Apply(rover.GetPrim())
    _bind_material(stage, rover.GetPrim(), "/World/Materials/Rover")


def _add_goal_marker(stage, elevation, resolution: float) -> None:
    row, col = elevation.shape[0] - 12, elevation.shape[1] - 12
    x, y, z = _grid_to_world(row, col, elevation, resolution)
    goal = UsdGeom.Sphere.Define(stage, "/World/PlantingGoal")
    goal.CreateRadiusAttr(0.35)
    goal.AddTranslateOp().Set(Gf.Vec3f(x, y, z + 0.45))
    _bind_material(stage, goal.GetPrim(), "/World/Materials/Goal")


def _add_light(stage) -> None:
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1200.0)
    sun.CreateAngleAttr(0.8)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 25.0))
    sky = UsdLux.DomeLight.Define(stage, "/World/SkyFill")
    sky.CreateIntensityAttr(180.0)
    sky.CreateColorAttr(Gf.Vec3f(0.68, 0.78, 0.95))


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


def _create_material(stage, path: str, color: tuple[float, float, float], roughness: float) -> None:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")


def _bind_material(stage, prim, material_path: str) -> None:
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _compute_heightfield_normals(elevation, resolution: float) -> list[Gf.Vec3f]:
    grad_y, grad_x = np.gradient(elevation.astype(np.float32), resolution)
    normals = np.dstack((-grad_x, -grad_y, np.ones_like(elevation, dtype=np.float32)))
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-6)
    return [
        Gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2]))
        for normal in normals.reshape(-1, 3)
    ]


def _set_vertex_colors(mesh: UsdGeom.Mesh, elevation) -> None:
    normalized = (elevation - float(elevation.min())) / max(float(np.ptp(elevation)), 1e-6)
    height, width = elevation.shape
    yy, xx = np.mgrid[0:height, 0:width]
    ripple = 0.5 + 0.5 * np.sin(xx * 0.23 + yy * 0.09)
    warmth = np.clip(0.68 + 0.16 * normalized + 0.08 * ripple, 0.0, 1.0)
    colors = [
        Gf.Vec3f(float(r), float(g), float(b))
        for r, g, b in np.column_stack(
            [
                warmth.reshape(-1),
                (0.50 + 0.10 * normalized + 0.04 * ripple).reshape(-1),
                (0.28 + 0.06 * normalized).reshape(-1),
            ]
        )
    ]
    display_color = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    display_color.Set(colors)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
