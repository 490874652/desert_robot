# Desert Robot AI Handoff - 2026-07-01

## 1. Project State

This repo is building a desert-tree-planting rover simulation loop:

```text
Isaac Sim desert scene
-> RGB/depth capture
-> depth to local heightfield
-> local costmap
-> A* local path
-> Isaac GUI rover path demo
```

Current focus is still Isaac Sim + Python algorithm prototyping. Do not move to ROS 2 or Isaac Lab yet unless the user explicitly asks.

## 2. Environment Rules

Use two Python environments:

```bash
# Normal algorithm, tests, plotting
cd /home/gx/work/desert_robot
source .venv/bin/activate

# Isaac Sim scripts
cd /home/gx/work/desert_robot
source /home/gx/env_isaacsim/bin/activate
```

Important:

```text
Do not install Isaac Sim packages into .venv.
Run isaac/*.py scripts with /home/gx/env_isaacsim/bin/python.
Run src/scripts/tests with /home/gx/work/desert_robot/.venv/bin/python.
```

## 3. Key Files

```text
isaac/desert_heightfield_scene.py
  Generates the USD desert scene. Current scene is larger and smoother than earlier versions.
  It includes sand vertex colors, ground reference lines, rocks, a hidden placeholder rover,
  a goal marker, lights, and camera.

isaac/capture_camera_frame.py
  Opens the USD scene and captures RGB, depth, and camera parameters.

scripts/build_local_heightfield_from_capture.py
  Converts Isaac depth + camera params into local_heightfield and local_costmap.

scripts/plan_local_path_from_costmap.py
  Runs A* on local_costmap and writes local_path.

scripts/live_perception_planning_demo.py
  Matplotlib dashboard for RGB/depth/heightfield/costmap/path.
  In non-interactive Agg backend it writes outputs/perception/live_dashboard.png.

isaac/live_rover_path_demo.py
  Isaac GUI rover demo. This is the main visualization script right now.
```

## 4. Current Outputs

Generated Isaac outputs:

```text
outputs/isaac/desert_heightfield_scene.usd
outputs/isaac/camera_rgb.png
outputs/isaac/camera_depth.npy
outputs/isaac/camera_depth_mm.png
outputs/isaac/camera_params.json
```

Generated perception outputs:

```text
outputs/perception/local_heightfield.npz
outputs/perception/local_heightfield.png
outputs/perception/local_costmap.npz
outputs/perception/local_costmap.png
outputs/perception/local_path.npz
outputs/perception/local_path.png
outputs/perception/live_dashboard.png
```

`outputs/perception` is not Isaac-specific by design. It currently comes from Isaac captures, but the same format should later accept real depth camera input plus calibration.

## 5. Full Pipeline Commands

Regenerate everything from scene to path:

```bash
cd /home/gx/work/desert_robot

# 1. Generate Isaac USD scene
/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py

# 2. Capture RGB/depth/camera params from Isaac scene
/home/gx/env_isaacsim/bin/python isaac/capture_camera_frame.py

# 3. Build local heightfield and costmap
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/build_local_heightfield_from_capture.py

# 4. Plan local A* path
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_path_from_costmap.py
```

Run the live rover demos:

```bash
# Demo 1: current A* endpoint-to-endpoint path
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1

# Demo 2: uphill path from low terrain to high terrain
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 2

# Optional loop
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1 --loop
```

Short validation runs:

```bash
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1 --frames 30
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 2 --frames 30
```

## 6. Rover Demo Details

`isaac/live_rover_path_demo.py` currently does kinematic visualization, not full PhysX vehicle simulation.

Current behavior:

```text
- /World/LiveRover is the single moving root node.
- Chassis, SensorMast, four Wheel_* prims, and four Hub_* prims are local children.
- The red arrow marks rover forward direction.
- The blue continuous ribbon marks the path and is lifted above terrain to avoid being hidden.
- Yellow disks are sparse path samples.
- Blue disk is start.
- Red flag is goal.
- Camera is reset to an overhead view centered on the current path.
- By default the rover stops at the end; --loop makes it loop.
```

Important implementation note:

```text
The rover root transform is written as a 4x4 matrix.
Do not go back to simple RotateXYZ yaw/pitch/roll unless there is a strong reason.
Earlier RotateXYZ versions caused apparent sideways/vertical direction errors.
```

The transform basis is:

```text
forward axis = path tangent
up axis      = normal estimated from four wheel contact samples
right axis   = cross product of forward/up
position     = path center XY + sampled ground height
```

## 7. Recently Fixed Issues

These were real problems observed by the user:

```text
1. Rover seemed to drive into sand.
   Cause: using local perception heightfield for visual Z created mismatch with true USD terrain.
   Fix: visual rover samples true /World/DesertTerrain.

2. Wheels and chassis separated.
   Cause: each part was updated independently in world coordinates.
   Fix: /World/LiveRover root moves; wheels/chassis are local child parts.

3. Rover direction did not match visible path.
   Cause: Euler RotateXYZ order was unreliable for path tangent + terrain tilt.
   Fix: root transform is explicit 4x4 matrix.

4. Rover appeared to jump randomly.
   Cause: path loop returned from end to start by default.
   Fix: default is stop at end; --loop is opt-in.

5. Blue path ribbon was hard to see.
   Cause: path ribbon was near z=0 and could be hidden by terrain.
   Fix: path visuals sample terrain height and are lifted above the surface.

6. Terrain/map was visually unclear.
   Fix: desert scene is larger, smoother, has more sand color variation and ground reference lines.
```

## 8. Current Validation

The latest working state was validated with:

```bash
/home/gx/work/desert_robot/.venv/bin/python -m ruff check isaac scripts src tests \
  --cache-dir /tmp/desert_robot_ruff_cache

/home/gx/work/desert_robot/.venv/bin/python -m pytest

/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py
/home/gx/env_isaacsim/bin/python isaac/capture_camera_frame.py
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/build_local_heightfield_from_capture.py
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_path_from_costmap.py

/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1 --frames 30
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 2 --frames 30
```

Last observed results:

```text
ruff: passed
pytest: 16 passed
demo 1: opened scene, loaded 106 points, exited cleanly
demo 2: opened scene, loaded 39 points, exited cleanly
```

## 9. Git State At Handoff

At the time this handoff was written, these files had local modifications:

```text
docs/project_overview.md
isaac/desert_heightfield_scene.py
isaac/live_rover_path_demo.py
```

This handoff file is new:

```text
docs/ai_handoff_2026-07-01.md
```

Generated outputs under `outputs/` may also have changed, but they are generated artifacts.

Suggested commit after user review:

```bash
git add docs/project_overview.md docs/ai_handoff_2026-07-01.md \
  isaac/desert_heightfield_scene.py isaac/live_rover_path_demo.py
git commit -m "Improve rover path demos and desert scene"
```

## 10. Recommended Next Steps

Best next step:

```text
Move from kinematic visualization toward a more physical rover:
1. Add wheel rotation animation based on traveled distance.
2. Add simple suspension visualization using per-wheel local Z offsets.
3. Then consider PhysX contact/raycast-based vehicle behavior.
```

Other useful next steps:

```text
- Add more demo paths: obstacle avoidance, ridge traverse, soft-sand region.
- Add keyboard/camera controls for switching demo path during runtime.
- Add a simple screenshot or video export command for user review.
- Make local map/path update while rover moves.
```

Defer for now:

```text
- ROS 2 integration.
- Isaac Lab / RL training.
- Full sand particle simulation.
```

## 11. User Preferences And Context

The user cares strongly about visual correctness:

```text
- Rover must visibly follow the path.
- Rover direction must match the path direction.
- Wheels and chassis must stay together.
- Ground should be visually clear enough to judge whether the rover is on the terrain.
- The user prefers practical incremental demos over abstract explanations.
```

When making future changes, validate visually in Isaac whenever possible, not only with pytest.

