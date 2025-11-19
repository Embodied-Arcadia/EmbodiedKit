# vln_data_generate

基于 NVIDIA Isaac Sim 的多场景导航路径生成与机器人导航数据采集流程。

整个 pipeline 分为三个层级：

- **Level 1 – main_controller.py**：批量遍历场景目录，对每个场景依次
  - 调用 `path_generator.py` 生成导航路径（NavMesh + 多条可行走路径），
  - 读取生成的 `generated_paths.json`，逐条调用 `robot_navigator.py` 进行机器人导航仿真与数据录制。
- **Level 2 – path_generator.py**：在单个 USD 场景中
  - 初始化 Isaac Sim、加载场景并调整缩放与高度，
  - 生成 NavMesh（支持 GPU/CPU、多次重试和内存保护），
  - 按约束条件随机采样起终点并生成多条“贴地”路径，
  - 将所有 episode 写入 `generated_paths.json`。
- **Level 3 – robot_navigator.py**：
  - 在指定 USD 场景中加载 G1 机器人（`g1_29dof_color_camera.usd`），
  - 让机器人沿输入路径导航并记录位姿轨迹、动作和相机视频，
  - 将结果保存到对应 episode 目录中。

---

## 1. 依赖与环境

- **仿真环境**：NVIDIA Isaac Sim 5.0.0
  - `isaac-sim5.0.0/python.sh` 提供运行环境和依赖（`omni.isaac.kit`、`isaacsim.core` 等）


> 建议始终通过 Isaac Sim 自带的 `python.sh` 启动 `path_generator.py` 和 `robot_navigator.py`，以确保依赖一致。

---

## 2. 目录结构

```text
path_generate_v2/
├── main_controller.py   # Level 1：批处理入口
├── path_generator.py    # Level 2：单场景路径生成
└── robot_navigator.py   # Level 3：机器人导航与数据录制
```

此外工程中还使用到：

- 机器人资产：`./assets/g1_29dof_color_camera.usd`
- 场景数据目录（默认）：`./assets/scenes` （暂无）
- 输出目录（默认）：`./data_output`

- 对于每个场景 `scene_name`：
  - 输出目录为：`<output_dir>/<scene_name>/`
  - 路径生成结果文件为：`<output_dir>/<scene_name>/generated_paths.json`
  - 单条 episode 的输出目录为：`<output_dir>/<scene_name>/episode_<episode_id>/`

可以通过命令行参数覆盖上述默认路径。

---

## 4. 各脚本功能与参数

### 4.1 main_controller.py（批处理入口 / Level 1）

**功能**：

- 遍历 `--scenes_dir` 下的所有子目录（按名字排序）。
- 在每个场景下：
  - 调用 Isaac Sim 的 Python（`<isaac_sim_path>/python.sh`）运行 `path_generator.py`，生成多条导航路径；
  - 读取生成的 `generated_paths.json`，按 episode 逐条调用 `robot_navigator.py`，在仿真中驱动机器人沿路径运动并保存数据。

**主要参数**：

```bash
python main_controller.py \
    --isaac_sim_path /path/to/isaac-sim5.0.0 \
    --scenes_dir /path/to/scenes_root \
    --output_dir /path/to/data_output \
    --num_paths_per_scene 20 \
    --run_mode run
```

- `--isaac_sim_path`（str）：Isaac Sim 根目录，内部会拼成 `<isaac_sim_path>/python.sh`。
- `--scenes_dir`（str）：包含多个场景子目录的根目录。
- `--output_dir`（str）：所有 scene/episode 输出根目录。
- `--num_paths_per_scene`（int）：每个场景希望生成的路径数量，会传给 `path_generator.py --num_paths`。
- `--run_mode`（`test` | `run`）：
  - `test`：只处理第一个场景，方便快速验证流程；
  - `run`：处理全部场景。

**输出**：

- 对每个场景：
  - 路径文件：`<output_dir>/<scene_name>/generated_paths.json`
  - 每条 episode：
    - `navigation_task.json`
    - `agent_trajectory.json`
    - `trajectory_video.mp4`（如有相机数据）

---

### 4.2 path_generator.py（路径生成 / Level 2）

**功能**：在单个 USD 场景中：

- 启动 Isaac Sim `SimulationApp`（headless 或 GUI）；
- 加载场景，缩放并抬起到 `z = 0` 附近；
- 关闭天花板、家具等的碰撞（减少 NavMesh 噪声）；
- 配置导航参数：
  - 代理半径 / 高度；
  - 最大坡度、最大台阶高度；
  - NavMesh 体积大小（约 550×550×550 m）；
- 多次尝试 GPU NavMesh baking，必要时自动切换到 CPU；
- 在地面高度范围内随机采样起点/终点，生成满足长度和“贴地”约束的多条路径；
- 对路径 z 值进行归一化（z-offset），保证所有点在统一的“地面高度”；
- 将所有 episode 写入一个 `generated_paths.json` 文件。

**命令行参数**：

```bash
<isaac_sim_path>/python.sh path_generator.py \
    --usd_path /path/to/scene.usd \
    --output_dir /path/to/data_output \
    --num_paths 20 \
    --headless \
    [--safe_mode] \
    [--visualize_navmesh]
```

- `--usd_path`（str, 必选）：输入场景 USD 文件路径。
- `--output_dir`（str, 必选）：输出根目录（脚本内部会在其中为该场景建立子目录）。
- `--num_paths`（int）：目标路径数量（默认 5）。
- `--safe_mode`（flag）：启用更保守的渲染/RTX 设置，避免 GPU 崩溃；
- `--visualize_navmesh`（flag）：在 GUI 模式下显示 NavMesh，可在 baking 前后进行人工检查；
- `--headless`（flag）：以无界面模式运行（建议在服务器或批处理时开启）。

**输出文件**（单个场景）：

- `scene_name = basename(dirname(usd_path))`
- 输出目录：`<output_dir>/<scene_name>/`
- 文件：
  - `generated_paths.json`
    - `episodes`: 列表，每个元素形如：
      - `episode_id`, `trajectory_id`, `scene_id`
      - `start_position`, `start_rotation`
      - `goals[0].position`, `goals[0].radius`
      - `reference_path`: 三维 waypoint 序列
      - `info`: 包含 `geodesic_distance`, `z_offset_applied`, `floor_z_range` 等调试信息

---

### 4.3 robot_navigator.py（导航仿真 / Level 3）

**功能**：

- 启动 Isaac Sim `SimulationApp`；
- 打开指定 USD 场景；
- 在 `/World/g1` 位置实例化 G1 机器人资产：`/home2/gmh/panxiaoran/g1_29dof_color_camera.usd`；
- 将相机挂在机器人 `base_link/torso_camera` 下，采集 RGB 帧；
- 使用 `SimplePathFollower` 控制机器人沿输入路径导航：
  - 根据当前姿态和下一 waypoint 计算线速度和角速度；
  - 调用 articulation API 或旧的 `apply_action` 接口执行动作；
- 记录每一步：位置、旋转和动作；
- 将轨迹和视频写入输出目录。

**命令行参数**：

```bash
<isaac_sim_path>/python.sh robot_navigator.py \
    --usd_path /path/to/scene.usd \
    --path_data_file /path/to/path_data.json \
    --output_dir /path/to/episode_output \
    --episode_id 0
```

- `--usd_path`（str, 必选）：输入场景 USD 文件，与路径生成使用的应保持一致。
- `--path_data_file`（str, 必选）：包含单个 episode 数据的 JSON 文件（`main_controller.py` 会从 `generated_paths.json` 中摘取一条写成临时文件）。
- `--output_dir`（str, 必选）：该 episode 的输出目录。
- `--episode_id`（int）：当前 episode 编号，仅用于记录。

**输出文件**：

- `navigation_task.json`：包含 episode 的起点、终点、参考路径、geodesic 距离等元信息。
- `agent_trajectory.json`：仿真过程中记录的机器人位姿和动作序列。
- `trajectory_video.mp4`：从相机视角录制的视频（如果相机初始化成功且有帧）。

---

## 5. 典型使用流程

### 5.1 单场景手动调试

1. **生成路径：**

   ```bash
   cd path/to/path_generate

   ISAAC_SIM=/path/to/isaac-sim5.0.0

   $ISAAC_SIM/python.sh path_generator.py \
       --usd_path /path/to/scene/start_result_navigation.usd \
       --output_dir /path/to/data_output \
       --num_paths 10 \
       --headless
   ```

2. **选择一条路径并运行导航（通常由 main_controller 自动完成）：**

   - 在 `<output_dir>/<scene_name>/generated_paths.json` 中选择一个 `episode`，或使用 `main_controller.py` 自动生成 `path_data.json` 并调用 `robot_navigator.py`。

   - 手动调用示例：

     ```bash
     EP_OUT=/path/to/data_output/<scene_name>/episode_0
     
     $ISAAC_SIM/python.sh robot_navigator.py \
         --usd_path /path/to/scene/start_result_navigation.usd \
         --path_data_file $EP_OUT/path_data.json \
         --output_dir $EP_OUT \
         --episode_id 0
     ```

### 5.2 全量批处理（推荐方式）

```bash
cd path/to/path_generate

python main_controller.py \
    --isaac_sim_path /path/to/isaac-sim5.0.0 \
    --scenes_dir /path/to/scenes_root \
    --output_dir /path/to/data_output \
    --num_paths_per_scene 20 \
    --run_mode run
```

- 若需要快速测试流程是否正常，可将 `--run_mode` 设置为 `test`，只处理第一个场景。

---

## 6. 常见注意事项

- **Isaac Sim 版本与路径**：
  - 请确认 `--isaac_sim_path` 与实际安装目录一致，并且其中存在 `python.sh`。
- **GPU 内存**：
  - NavMesh baking 对 GPU 内存有一定要求；当检测到 “Out of GPU memory” 等错误时，脚本会自动退回到 CPU baking，但仍可能较慢。
- **场景碰撞体**：
  - 如果 USD 场景缺少启用碰撞的几何体，NavMesh 可能无法生成或生成为空，脚本会给出相应的 Fatal Error 提示。
- **地面高度与多层结构**：
  - 代码通过采样 NavMesh z 值、聚类地面高度并加 z-offset 的方式，过滤楼梯/家具等非地面区域；在极端场景（如多层楼或地形异常）可能需要调整脚本中的阈值。
- **机器人资产路径**：
  - 当前 `robot_navigator.py` 中的 G1 USD 路径是硬编码的，如路径或文件名改变，需要同步修改代码。

---
