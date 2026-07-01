# Desert Robot 常用命令手册

更新时间：2026-07-01

## 1. 最重要的一点

运行本项目脚本前，先进入项目文件夹：

```bash
cd /home/gx/work/desert_robot
```

如果你在 `~` 目录下直接运行：

```bash
python scripts/generate_local_subgoals.py
```

系统会去找：

```text
/home/gx/scripts/generate_local_subgoals.py
```

这就是你刚才报错的原因。正确方式是先 `cd`：

```bash
cd /home/gx/work/desert_robot
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/generate_local_subgoals.py
```

## 2. 两个 Python 环境

这个项目有两套 Python 环境，不要混用。

### 普通算法环境

用于运行：

```text
scripts/
src/
tests/
matplotlib 画图
路径规划
局部目标点生成
```

进入方式：

```bash
cd /home/gx/work/desert_robot
source .venv/bin/activate
```

也可以不用 activate，直接用完整 Python 路径：

```bash
/home/gx/work/desert_robot/.venv/bin/python scripts/generate_local_subgoals.py
```

### Isaac Sim 环境

用于运行：

```text
isaac/
生成 USD 场景
采集 RGB/depth
打开 Isaac GUI 小车演示
```

进入方式：

```bash
cd /home/gx/work/desert_robot
source /home/gx/env_isaacsim/bin/activate
```

也可以不用 activate，直接用完整 Python 路径：

```bash
/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py
```

注意：不要把 Isaac Sim 包安装进项目 `.venv`。

## 3. 推荐运行方式

最稳的方式是每条命令都写完整路径，并且先进入项目目录：

```bash
cd /home/gx/work/desert_robot
```

普通算法脚本用：

```bash
/home/gx/work/desert_robot/.venv/bin/python
```

Isaac 脚本用：

```bash
/home/gx/env_isaacsim/bin/python
```

## 4. 完整流水线

从生成沙漠场景到局部目标点可视化，按这个顺序运行。

### 1. 进入项目目录

```bash
cd /home/gx/work/desert_robot
```

### 2. 生成 Isaac 沙漠 USD 场景

```bash
/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py
```

输出：

```text
outputs/isaac/desert_heightfield_scene.usd
```

### 3. 采集 RGB/depth/camera 参数

```bash
/home/gx/env_isaacsim/bin/python isaac/capture_camera_frame.py
```

输出：

```text
outputs/isaac/camera_rgb.png
outputs/isaac/camera_depth.npy
outputs/isaac/camera_depth_mm.png
outputs/isaac/camera_params.json
```

### 4. 从 depth 生成局部 heightfield 和 costmap

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/build_local_heightfield_from_capture.py
```

输出：

```text
outputs/perception/local_heightfield.npz
outputs/perception/local_heightfield.png
outputs/perception/local_costmap.npz
outputs/perception/local_costmap.png
```

### 5. 在 costmap 上规划 A* 路径

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_path_from_costmap.py
```

输出：

```text
outputs/perception/local_path.npz
outputs/perception/local_path.png
```

### 6. 生成局部目标点 subgoals

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/generate_local_subgoals.py
```

输出：

```text
outputs/perception/local_subgoals.npz
outputs/perception/local_subgoals.png
```

### 7. 生成局部目标序列

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_subgoal_sequence.py
```

输出：

```text
outputs/perception/local_subgoal_sequence.npz
outputs/perception/local_subgoal_sequence.png
```

## 5. 常用可视化检查

### 查看局部高度图

打开：

```text
outputs/perception/local_heightfield.png
```

检查点：

```text
地形是否连续
未知区域是否合理
高度图是否明显错位
```

### 查看履带车 costmap

打开：

```text
outputs/perception/local_costmap.png
```

检查点：

```text
灰色区域是否为不可通行区域
高代价区域是否对应陡坡/软沙/障碍
履带车可通行性分级是否合理
```

### 查看 A* 路径

打开：

```text
outputs/perception/local_path.png
```

检查点：

```text
路径是否避开灰色 blocked 区域
路径是否明显绕开障碍
路径是否过度贴近不可通行边界
```

### 查看局部目标点

打开：

```text
outputs/perception/local_subgoals.png
```

图中标记：

```text
绿色星号：当前 start
绿色箭头：局部前进方向
青色圆点：safe_frontier，安全前进点
橙色三角：climb_entry，可爬坡入口点
白色方块：gap，障碍/不可通行区域之间的通道点
数字编号：候选点优先级，数字越小越优先
灰色区域：不可通行/未观测/blocked
```

检查点：

```text
候选点是否主要在车头方向
gap 是否真的落在通道附近
climb_entry 是否靠近坡度入口，而不是直接落在危险陡坡深处
safe_frontier 是否在视野前方边界附近
```

### 查看局部目标序列

打开：

```text
outputs/perception/local_subgoal_sequence.png
```

图中标记：

```text
绿色星号：当前 start
绿色箭头：初始局部前进方向
蓝色折线：依次连接 subgoals 的规划路径
数字编号：目标序列顺序
青色圆点：safe_frontier
橙色三角：climb_entry
白色方块：gap
灰色区域：不可通行/未观测/blocked
```

检查点：

```text
目标序列是否整体朝车头方向推进
每一段蓝色路径是否避开 blocked 区域
序列是否在局部视野内合理利用 gap 或 climb_entry
目标点之间是否太近或来回摆动
```

## 6. Isaac GUI 小车演示

进入项目目录：

```bash
cd /home/gx/work/desert_robot
```

运行 demo 1：

```bash
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1
```

运行 demo 2：

```bash
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 2
```

短测试，只跑 30 帧：

```bash
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1 --frames 30
```

循环播放：

```bash
/home/gx/env_isaacsim/bin/python isaac/live_rover_path_demo.py --demo 1 --loop
```

## 7. 运行测试

进入项目目录：

```bash
cd /home/gx/work/desert_robot
```

运行全部测试：

```bash
/home/gx/work/desert_robot/.venv/bin/python -m pytest
```

运行代码风格检查：

```bash
/home/gx/work/desert_robot/.venv/bin/python -m ruff check isaac scripts src tests \
  --cache-dir /tmp/desert_robot_ruff_cache
```

## 8. Git 常用命令

进入项目目录：

```bash
cd /home/gx/work/desert_robot
```

查看当前改动：

```bash
git status
```

查看简短改动：

```bash
git status --short
```

查看具体修改内容：

```bash
git diff
```

添加所有改动：

```bash
git add .
```

提交改动：

```bash
git commit -m "Add tracked rover subgoal planning"
```

上传到远程仓库：

```bash
git push
```

如果你只想添加某几个文件，例如：

```bash
git add src/desert_robot/planning/subgoals.py scripts/generate_local_subgoals.py
```

再提交：

```bash
git commit -m "Add local subgoal generation"
```

## 9. 常见错误

### 找不到脚本文件

错误类似：

```text
can't open file '/home/gx/scripts/generate_local_subgoals.py'
```

原因：你没有先进入项目目录。

解决：

```bash
cd /home/gx/work/desert_robot
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/generate_local_subgoals.py
```

### Isaac 脚本导入失败

可能原因：用了项目 `.venv` 跑 Isaac 脚本。

错误方式：

```bash
/home/gx/work/desert_robot/.venv/bin/python isaac/desert_heightfield_scene.py
```

正确方式：

```bash
/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py
```

### 普通脚本找不到输出文件

例如生成 subgoals 前，需要先有：

```text
outputs/perception/local_costmap.npz
```

如果没有，先运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/build_local_heightfield_from_capture.py
```

## 10. 最常用命令清单

进入项目：

```bash
cd /home/gx/work/desert_robot
```

生成场景：

```bash
/home/gx/env_isaacsim/bin/python isaac/desert_heightfield_scene.py
```

采集相机：

```bash
/home/gx/env_isaacsim/bin/python isaac/capture_camera_frame.py
```

生成 costmap：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/build_local_heightfield_from_capture.py
```

规划路径：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_path_from_costmap.py
```

生成局部目标点：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/generate_local_subgoals.py
```

生成局部目标序列：

```bash
MPLCONFIGDIR=/tmp/matplotlib-desert-robot \
  /home/gx/work/desert_robot/.venv/bin/python scripts/plan_local_subgoal_sequence.py
```

运行测试：

```bash
/home/gx/work/desert_robot/.venv/bin/python -m pytest
```

查看 git 状态：

```bash
git status --short
```
