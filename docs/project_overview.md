# Desert Robot Project Overview

更新时间：2026-06-28

## 1. 项目目标

本项目用于构建“沙漠树苗自动种植小车”的仿真与算法验证闭环：

```text
Isaac Sim 沙漠场景
-> 相机 RGB / depth 采集
-> camera_params 导出
-> depth 反投影局部 heightfield
-> 局部 cost map
-> A* 局部路径规划
```

当前阶段重点是 Isaac Sim + Python 算法闭环，不是 ROS 2，也不是强化学习。

## 2. 项目框架

```text
desert_robot/
├── isaac/
│   ├── hello_isaac_scene.py
│   ├── desert_heightfield_scene.py
│   ├── view_usd_scene.py
│   ├── capture_camera_frame.py
│   └── live_rover_path_demo.py
├── scripts/
│   ├── check_env.py
│   ├── run_minimal_planning_demo.py
│   ├── build_local_heightfield_from_capture.py
│   ├── plan_local_path_from_costmap.py
│   └── live_perception_planning_demo.py
├── src/desert_robot/
│   ├── maps/
│   ├── perception/
│   ├── planning/
│   ├── utils/
│   ├── control/
│   ├── envs/
│   └── learning/
├── tests/
├── outputs/
└── docs/
```

## 3. 环境使用

普通算法环境：

```bash
cd /home/gx/work/desert_robot
source .venv/bin/activate
```

Isaac Sim 环境：

```bash
cd /home/gx/work/desert_robot
source /home/gx/env_isaacsim/bin/activate
```

注意：不要把 Isaac Sim 包安装进 `.venv`。Isaac 脚本用 `/home/gx/env_isaacsim`，普通算法、测试、绘图脚本用项目 `.venv`。

## 4. Isaac 场景与查看方法

生成沙漠 USD 场景：

```bash
source /home/gx/env_isaacsim/bin/activate
python isaac/desert_heightfield_scene.py
```

输出：

```text
outputs/isaac/desert_heightfield_scene.usd
```

在线用 Isaac GUI 查看场景：

```bash
source /home/gx/env_isaacsim/bin/activate
python isaac/view_usd_scene.py
```

默认打开：

```text
outputs/isaac/desert_heightfield_scene.usd
```

打开指定 USD：

```bash
python isaac/view_usd_scene.py outputs/isaac/hello_isaac_scene.usd
```

不要直接运行：

```bash
isaacsim outputs/isaac/desert_heightfield_scene.usd
```

因为 `isaacsim` 的第一个位置参数期望 `.kit` experience 文件，不是 `.usd` 场景。

## 5. 当前流水线

完整运行顺序：

```bash
# 1. 生成 Isaac 沙漠场景
source /home/gx/env_isaacsim/bin/activate
python isaac/desert_heightfield_scene.py

# 2. 采集 RGB / depth / camera_params
python isaac/capture_camera_frame.py

# 3. 构建局部 heightfield 和 cost map
source /home/gx/work/desert_robot/.venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib-desert-robot python scripts/build_local_heightfield_from_capture.py

# 4. 在局部 cost map 上规划路径
MPLCONFIGDIR=/tmp/matplotlib-desert-robot python scripts/plan_local_path_from_costmap.py

# 5. 打开/保存实时感知规划 dashboard
python scripts/live_perception_planning_demo.py --watch
MPLCONFIGDIR=/tmp/matplotlib-desert-robot python scripts/live_perception_planning_demo.py --save-snapshot

# 6. 在 Isaac GUI 里查看小车沿规划路径运动
source /home/gx/env_isaacsim/bin/activate
python isaac/live_rover_path_demo.py
```

## 6. 输出结果说明

### outputs/isaac

`outputs/isaac` 是 Isaac Sim 直接生成或采集的结果：

```text
desert_heightfield_scene.usd
camera_rgb.png
camera_depth.npy
camera_depth_mm.png
camera_params.json
```

含义：

```text
camera_rgb.png        Isaac 相机 RGB 图像
camera_depth.npy      原始 depth 数组，单位米，可能含 inf
camera_depth_mm.png   16-bit depth PNG，单位毫米，非有限值写为 0
camera_params.json    相机内参、外参、分辨率、裁剪范围
```

### outputs/perception

`outputs/perception` 当前是从 Isaac 采集结果派生出来的 perception 输出，但它本身不是 Isaac 专属格式。

也就是说：

```text
当前来源：Isaac camera_depth.npy + camera_params.json
逻辑归属：感知算法输出
未来来源：真实相机 depth + 相机标定参数也可以复用同一套流程
```

输出：

```text
local_heightfield.npz
local_heightfield.png
local_costmap.npz
local_costmap.png
local_path.npz
local_path.png
live_dashboard.png
```

含义：

```text
local_heightfield.npz   局部高度图、观测 mask、origin_xy、resolution
local_heightfield.png   局部高度图预览
local_costmap.npz       局部代价地图、坡度、可通行 mask、origin_xy、resolution
local_costmap.png       局部代价地图预览
local_path.npz          A* 路径、start、goal、总代价、访问格子数
local_path.png          路径叠加在局部 cost map 上的预览
live_dashboard.png      RGB、depth、local heightfield、local costmap/path 四宫格预览
```

## 7. 主要函数说明

### src/desert_robot/maps/heightfield.py

`generate_desert_heightfield(...)`

生成程序化沙漠 heightfield，包含：

```text
elevation
obstacle_mask
soil_looseness
bearing_capacity
resolution
```

`_generate_soil_looseness(...)`

内部函数，生成松软沙土分布，用于 slip/sinkage 风险估计。

### src/desert_robot/maps/costmap.py

`compute_slope(elevation, resolution)`

从高度图计算坡度幅值。

`build_costmap(...)`

把高度、障碍、松软度、承载力合成多风险 cost map，输出：

```text
cost
slope
slip_risk
sinkage_risk
planting_suitability
traversable_mask
```

`_optional_layer(...)`

内部函数，用默认值补齐可选风险图层。

### src/desert_robot/planning/astar.py

`astar(costmap, start, goal, obstacle_cost=1_000_000.0)`

在二维 cost grid 上执行 8 邻接 A*。返回：

```text
path
total_cost
visited_count
found
```

`_validate_point(...)`

内部函数，检查 start/goal 是否越界。

`_in_bounds(...)`

内部函数，检查格子是否在地图内。

`_heuristic(...)`

内部函数，A* 欧氏距离启发式。

`_reconstruct_path(...)`

内部函数，从 came_from 字典回溯路径。

### src/desert_robot/utils/terrain_mesh.py

`heightfield_to_triangle_mesh(elevation, resolution, center=True)`

把 2D heightfield 转成 USD/渲染可用的 triangle mesh 数据：

```text
vertices
face_vertex_counts
face_vertex_indices
```

### src/desert_robot/perception/depth_to_heightfield.py

`camera_model_from_params(params)`

读取 `camera_params.json` 风格的字典，构建 `CameraModel`。

`depth_to_world_points(depth, camera, max_depth=None)`

把 Isaac depth 图反投影成世界坐标点云。当前按 USD 相机约定处理：相机朝本地 `-Z` 方向看。

`world_points_to_heightfield(points, resolution, bounds_xy=None, statistic="median")`

把世界坐标点云按 XY 栅格化成局部 heightfield。

`depth_to_local_heightfield(...)`

组合函数：depth -> world points -> local heightfield。

`_aggregate(values, statistic)`

内部函数，同一栅格多个点时做 median / mean / min / max 聚合。

## 8. Isaac 脚本函数说明

### isaac/desert_heightfield_scene.py

`main()`

创建 Isaac USD stage，生成程序化沙漠地形，加入障碍、车辆占位体、目标点、灯光和相机，并保存 USD。

`_create_world(stage)`

设置 Z-up、米制单位、World 根节点、PhysicsScene 和材质。

`_add_desert_mesh(stage, elevation, resolution)`

把 heightfield 转成 USD mesh，加入平滑法线和沙色顶点色。

`_add_obstacle_markers(...)`

把 obstacle mask 中的采样点显示成岩石占位体。

`_add_rover_placeholder(...)`

添加小车占位块。

`_add_goal_marker(...)`

添加种植目标点。

`_add_light(stage)`

添加太阳光和天空补光。

`_add_camera(stage)`

添加 `/World/Camera`。

`_grid_to_world(...)`

把 heightfield 行列坐标转为世界坐标。

`_create_material(...)`

创建 USD PreviewSurface 材质。

`_bind_material(...)`

把材质绑定到 prim。

`_compute_heightfield_normals(...)`

根据高度梯度计算平滑顶点法线，减少地形棱角感。

`_set_vertex_colors(...)`

为沙地设置程序化顶点色，让 RGB 图像有沙色变化。

### isaac/capture_camera_frame.py

`main()`

打开 USD 场景，从相机采集 RGB、depth，并导出相机参数。

`_extract_array(...)`

从 Replicator annotator 输出中取 numpy 数组。

`_save_rgb(...)`

保存 RGB PNG。

`_save_depth(...)`

保存 depth NPY 和 16-bit depth PNG。

`_save_camera_params(...)`

保存相机内参、外参、分辨率和裁剪范围。

`_matrix_to_list(...)`

把 USD `Gf.Matrix4d` 转成 JSON 可写的 list。

### isaac/view_usd_scene.py

`parse_args()`

解析要打开的 USD 路径。

`main()`

启动 Isaac GUI 并持续 update，用于在线查看 USD 场景。

### isaac/live_rover_path_demo.py

`parse_args()`

解析 USD 场景、`local_path.npz`、`local_heightfield.npz`、小车速度和运行帧数。

`main()`

启动 Isaac GUI，打开沙漠 USD 场景，读取局部规划路径，把局部路径坐标转换到世界坐标，在场景中绘制路径点，并让 `/World/RoverPlaceholder` 沿路径循环运动。

常用方式：

```bash
source /home/gx/env_isaacsim/bin/activate
python isaac/live_rover_path_demo.py
```

短时间验证：

```bash
python isaac/live_rover_path_demo.py --frames 120
```

输出结果：

```text
Isaac GUI 中可以看到沙漠场景、小车占位体、起点/终点/路径点，以及小车沿 A* 路径移动。
终端会打印打开的 USD、路径点数量和路径长度。
```

`_wait_for_stage()`

等待 Isaac 把 USD stage 加载完成。

`_load_world_path()`

读取 `local_path.npz` 和 `local_heightfield.npz`，把路径从局部 costmap 行列坐标转换成 Isaac 世界坐标。

`_safe_elevation(...)`

为路径点取得地形高度；如果局部 heightfield 某个格子没有有效高度，用有效高度中位数兜底。

`_draw_path_markers(...)`

在 Isaac 场景中绘制起点、终点和若干白色路径点。

`_draw_sphere(...)`

创建路径标记球体，并设置位置、半径和颜色。

`_set_active_camera()`

把 GUI 视口切换到 `/World/Camera`，方便打开后直接看到当前沙漠场景。

`_set_translate(...)`

设置 prim 的平移位置；用于移动小车占位体和摆放路径标记。

`_cumulative_distance(...)`

计算路径上每个点对应的累计距离，用于按速度插值运动。

`_interpolate_path(...)`

根据目标行驶距离，在相邻路径点之间插值，得到当前小车位置。

## 9. scripts 函数说明

### scripts/run_minimal_planning_demo.py

`main()`

生成程序化地形，构建 cost map，跑 A*，保存最小算法闭环可视化。

`plot_demo(...)`

绘制 heightfield、soil looseness、sinkage risk、cost map 和路径。

`summarize_path(...)`

统计路径长度、平均 cost、最大坡度、风险均值等指标。

### scripts/build_local_heightfield_from_capture.py

`main()`

读取 Isaac depth 和 camera params，生成：

```text
local_heightfield.npz/png
local_costmap.npz/png
```

`_plot_heightfield(...)`

绘制局部 heightfield。

`_fill_unknown_elevation(...)`

用观测区域中位数填充未知高度，便于 cost map 计算坡度；未知区域仍会被标记为不可通行。

`_plot_costmap(...)`

绘制局部 cost map。

### scripts/plan_local_path_from_costmap.py

`main()`

读取 `local_costmap.npz`，自动选择 start/goal，运行 A*，输出：

```text
local_path.npz
local_path.png
```

`_parse_point(...)`

解析 `row,col` 格式的手动起点/终点。

`_default_start(...)`

默认起点：地图底部中心附近。

`_default_goal(...)`

默认终点：地图顶部中心附近。

`_nearest_traversable(...)`

找到离目标点最近的可通行格子。

`_nearest_in_mask(...)`

在指定 mask 内找最近格子。

`_largest_connected_component(...)`

找最大可通行连通区域，避免默认路径落在被未知区域切断的小块里。

`_nearest_reachable(...)`

在起点可达连通区域内找离目标最近的格子。

`_reachable_mask(...)`

从起点 flood fill，得到可达区域。

`_plot_path(...)`

把 A* 路径叠加到局部 cost map 图上。

### scripts/live_perception_planning_demo.py

`main()`

启动实时/准实时 dashboard。默认读取 `outputs/isaac` 中的 RGB、depth、camera params，并现场生成 local heightfield、cost map 和 A* path。

常用方式：

```bash
# 交互窗口，持续刷新
python scripts/live_perception_planning_demo.py --watch

# 无 GUI 环境下保存一张 dashboard
MPLCONFIGDIR=/tmp/matplotlib-desert-robot python scripts/live_perception_planning_demo.py --save-snapshot
```

`parse_args()`

解析 capture 目录、分辨率、最大 depth、刷新频率、watch/snapshot 参数。

`_run_watch(...)`

循环刷新 dashboard，用于观察地图和路径变化。

`_run_once(...)`

单次生成 dashboard，可选择保存 PNG 或打开 Matplotlib 窗口。

`_build_dashboard_frame(...)`

读取 RGB/depth/camera params，重建 local heightfield、cost map，并运行 A*。

`_draw_dashboard(...)`

绘制四宫格：

```text
Isaac RGB
Isaac depth
local heightfield
local costmap + A* path
```

`_draw_path(...)`

在 local costmap 上绘制路径，并在 watch 模式中按时间逐步显示路径。

`_reset_colorbars(...)`

刷新图像时清理旧 colorbar，避免 UI 越画越乱。

`_fill_unknown_elevation(...)`

用观测区域中位数填充未知高度，便于计算坡度；未知区域仍在 costmap 中保持阻塞。

`_default_start(...)` / `_default_goal(...)`

给局部规划选择默认起点和目标：底部中心到顶部中心。

`_nearest_in_mask(...)`

在指定 mask 内寻找最近格子。

`_largest_connected_component(...)`

选取最大可通行连通区域，避免默认路径落入孤立小区域。

`_reachable_mask(...)`

从起点 flood fill 得到可达区域。

`_animated_path_fraction()`

生成 0 到 1 的循环进度，用于 watch 模式下逐步显示路径。

## 10. 实时查看与性能监控

### 查看场景和地图

Isaac GUI 查看三维场景：

```bash
source /home/gx/env_isaacsim/bin/activate
python isaac/view_usd_scene.py
```

Isaac GUI 查看小车沿规划路径运动：

```bash
source /home/gx/env_isaacsim/bin/activate
python isaac/live_rover_path_demo.py
```

查看感知/规划 dashboard：

```bash
source /home/gx/work/desert_robot/.venv/bin/activate
python scripts/live_perception_planning_demo.py --watch
```

如果当前 Matplotlib 后端是 `Agg`，它不能弹出交互窗口。脚本会自动退化为循环刷新：

```text
outputs/perception/live_dashboard.png
```

如果没有图形窗口，保存一张静态 dashboard：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot python scripts/live_perception_planning_demo.py --save-snapshot
```

输出：

```text
outputs/perception/live_dashboard.png
```

### 查看 GPU 占用

最常用：

```bash
watch -n 1 nvidia-smi
```

更细的实时指标：

```bash
nvidia-smi dmon -s pucvmet
```

含义大致是：

```text
p  power
u  utilization
c  clocks
v  memory / video
m  memory
e  ecc / errors
t  temperature
```

如果系统安装了 `nvtop`，可以用：

```bash
nvtop
```

它比 `nvidia-smi` 更适合长期观察 GPU 使用率、显存、进程。

### 查看 CPU / 内存占用

交互式：

```bash
htop
```

如果安装了 `btop`：

```bash
btop
```

查看某个 Python / Isaac 进程：

```bash
ps aux | grep -E "isaac|python"
```

按进程观察 CPU 和内存：

```bash
top -p <PID>
```

### 推荐监控方式

开三个终端：

```text
终端 1：运行 Isaac 或 dashboard
终端 2：watch -n 1 nvidia-smi
终端 3：htop
```

这样可以同时看：

```text
GPU 利用率
显存占用
CPU 占用
内存占用
Isaac / Python 是否卡住
```

当前 Isaac 日志里出现过：

```text
CPU performance profile is set to powersave
```

这表示 CPU 当前偏省电，会影响仿真性能。后续做长时间仿真时，可以考虑切换到 performance governor。

## 11. 沙子流动的后续实现路线

短期不建议直接做完整颗粒仿真。建议分三阶段：

```text
1. Heightfield 形变：车轮经过后改变局部 elevation、sinkage、slip risk。
2. 局部流沙/滑塌：坡度过大时做 erosion/deposition，让沙子向低处搬运。
3. 局部高保真颗粒或 MPM：只在车轮附近小范围启用，用于特写或高精实验。
```

这样能保持主循环轻量，同时逐步接近真实沙地行为。
