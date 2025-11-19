# VLA训练数据生成器

使用Franka机械臂和RRT路径规划生成VLA (Vision-Language-Action) 训练数据的完整系统。


## 功能特点

- ✅ **Franka机械臂支持**: 使用标准的Franka Panda机械臂
- ✅ **RRT路径规划**: 基于Isaac Sim的Lula RRT算法生成无碰撞路径（26个参数完整配置）
- ✅ **上帝视角采集**: 从场景上方采集RGB图像
- ✅ **自然语言指令**: 支持中英文任务指令
- ✅ **RLDS格式输出**: 标准的Reinforcement Learning Datasets格式
- ✅ **完整轨迹记录**: 记录关节位置、末端执行器位姿、图像和动作
- ✅ **GPU加速**: 支持CUDA加速的RRT规划

## 系统架构

```
vla_path_generate/
├── franka_rrt_controller.py    # Franka机械臂RRT控制器
├── rlds_writer.py               # RLDS数据集写入器
├── vla_data_generator.py        # 主数据生成器
├── main_generator.py            # 主控制脚本
├── config_example.json          # 配置文件示例
├── configs/                     # 配置文件目录
│   ├── franka_description.yaml  # 机器人描述
│   ├── franka.urdf              # Franka URDF
│   └── franka_rrt_config.yaml   # RRT配置
└── README.md                    # 本文件
```

## 环境依赖

### 必需软件

- **NVIDIA Isaac Sim 4.5.0+**: 物理仿真和机器人控制
- **Python 3.8+**: 编程语言
- **CUDA 11.0+**: GPU加速

### Python依赖

```bash
# Isaac Sim自带的依赖（通常已安装）
- isaacsim
- omni.isaac.core
- omni.isaac.motion_generation

# 额外依赖
- numpy>=1.19.0
- scipy>=1.5.0
- tensorflow>=2.8.0
- tensorflow-datasets>=4.5.0
- Pillow>=8.0.0
```

## 快速开始

### 1. 安装依赖

```bash
# 确保已安装Isaac Sim
# 激活Isaac Sim的Python环境
source ~/.local/share/ov/pkg/isaac-sim-*/setup_python_env.sh

# 安装额外依赖
pip install tensorflow tensorflow-datasets scipy Pillow
```

### 2. 准备Franka URDF

需要提供Franka Panda的URDF文件。可以从以下来源获取：

- [Franka Robotics官方](https://github.com/frankaemika/franka_ros)
- Isaac Sim内置资源

将URDF文件放置在 `configs/franka.urdf`。

### 3. 设置配置文件

```bash
# 方式1：使用脚本自动设置
python main_generator.py --setup --create-template

# 方式2：手动复制示例配置
cp config_example.json config.json
```

### 4. 编辑配置文件

编辑 `config.json`，根据需要修改：

```json
{
  "dataset_name": "vla_franka_manipulation",
  "output_dir": "./vla_output",
  "num_episodes": 10,
  
  "robot_description_path": "./configs/franka_description.yaml",
  "urdf_path": "./configs/franka.urdf",
  "rrt_config_path": "./configs/franka_rrt_config.yaml",
  
  "objects": [
    {
      "name": "red_cube",
      "type": "cuboid",
      "position": [0.5, 0.0, 0.05],
      "size": [0.05, 0.05, 0.05],
      "color": [1.0, 0.0, 0.0]
    }
  ],
  
  "tasks": [
    {
      "instruction": "拿起红色方块放到另一处",
      "target_object": "red_cube",
      "target_position": [0.3, 0.3, 0.05]
    }
  ]
}
```

### 5. 生成数据（三种方法）

**方法1: 使用Python启动器（最简单，推荐）**
```bash
python run_simple.py --config config_example.json
```

**方法2: 使用Shell脚本**
```bash
bash run_with_isaac.sh --config config_example.json
```

**方法3: 手动激活环境**
```bash
# 激活Isaac Sim环境
source ~/.local/share/ov/pkg/isaac-sim-4.5.0/setup_python_env.sh

# 验证配置
python main_generator.py --validate --config config.json

# 开始生成
python main_generator.py --generate --config config.json
```

### 6. 可视化生成的轨迹

```bash
# 查看单个轨迹
python visualize_trajectory.py --data-dir ./vla_output --episode-idx 0

# 生成完整分析报告
python visualize_trajectory.py --data-dir ./vla_output --generate-report --output-dir ./analysis

# 对比多个episodes
python visualize_trajectory.py --data-dir ./vla_output --compare-episodes 5
```

**生成的可视化内容**:
- 🎯 3D末端执行器轨迹（起点、终点、路径）
- 📊 7个关节角度变化曲线
- 📈 XYZ位置分量时间序列
- 📝 任务统计信息（步数、路径长度、效率等）

更详细的可视化用法说明可参考 `visualize_trajectory.py` 脚本中的参数和注释。

## 使用指南

### 主控制脚本

`main_generator.py` 提供了便捷的命令行接口：

```bash
# 查看帮助
python main_generator.py --help

# 设置配置文件
python main_generator.py --setup --config-dir ./configs

# 创建配置模板
python main_generator.py --create-template --output my_config.json

# 验证配置
python main_generator.py --validate --config my_config.json

# 生成数据
python main_generator.py --generate --config my_config.json

# 一键设置并生成
python main_generator.py --setup --create-template --generate
```

### 配置文件说明

#### 基本配置

- `dataset_name`: 数据集名称
- `dataset_description`: 数据集描述
- `output_dir`: 输出目录
- `num_episodes`: 生成的episode数量
- `shard_size`: 每个TFRecord shard的episode数量

#### 机器人配置

- `robot_description_path`: 机器人描述YAML文件路径
- `urdf_path`: Franka URDF文件路径
- `rrt_config_path`: RRT配置YAML文件路径
- `robot_asset_path`: 机器人USD资产路径（可选）
- `robot_start_position`: 机器人起始位置 [x, y, z]

#### 场景配置

- `scene_usd_path`: 场景USD文件路径（可选，null则创建空场景）
- `objects`: 场景物体列表
  - `name`: 物体名称
  - `type`: 物体类型（cuboid, sphere）
  - `position`: 位置 [x, y, z]
  - `size`: 尺寸 [长, 宽, 高]（仅cuboid）
  - `radius`: 半径（仅sphere）
  - `color`: RGB颜色 [r, g, b]

#### 任务配置

- `tasks`: 任务列表
  - `instruction`: 自然语言指令（支持中英文）
  - `target_object`: 目标物体名称
  - `target_position`: 目标位置 [x, y, z]

#### 相机配置

- `camera`: 相机参数
  - `position`: 位置 [x, y, z]
  - `orientation`: 朝向四元数 [x, y, z, w]
  - `resolution`: 分辨率 [width, height]

### RRT参数调优

编辑 `configs/franka_rrt_config.yaml` 来调整RRT算法参数：

```yaml
rrt_config:
  seed: 42                    # 随机种子
  step_size: 0.1              # 步长
  max_iterations: 10000       # 最大迭代次数
  
  # C空间规划参数
  c_space_planning_params:
    exploration_fraction: 0.8  # 探索比例
  
  # 任务空间规划参数
  task_space_planning_params:
    translation_target_zone_tolerance: 0.01  # 位置容差
    orientation_target_zone_tolerance: 0.05  # 姿态容差
    # ... 更多参数见配置文件
```

## 数据格式

### RLDS格式

生成的数据遵循标准的RLDS格式：

```python
{
  "episode_id": int,
  "steps": [
    {
      "observation": {
        "image": np.ndarray,              # (H, W, 3) RGB图像
        "joint_positions": np.ndarray,    # (7,) 关节位置
        "ee_position": np.ndarray,        # (3,) 末端执行器位置
        "ee_orientation": np.ndarray,     # (4,) 末端执行器姿态（四元数）
        "instruction": str                # 任务指令
      },
      "action": np.ndarray,               # (7,) 动作（关节位置增量）
      "reward": float,                    # 奖励
      "is_first": bool,                   # 是否第一步
      "is_last": bool,                    # 是否最后一步
      "is_terminal": bool                 # 是否终止状态
    },
    ...
  ],
  "metadata": {
    "instruction": str,
    "task_success": bool,
    "num_steps": int,
    ...
  }
}
```

### 输出文件

数据生成后会在输出目录创建以下文件：

```
vla_output/
├── tfrecords/                          # TFRecord格式数据
│   ├── vla_franka_manipulation-00000-of-00001.tfrecord
│   └── ...
└── vla_franka_manipulation.json        # JSON格式元数据（用于调试）
```

## RLDS格式转换（从 vla_output 到 vla_rlds）

在完成数据生成之后，可以将 `vla_output/` 中的原始数据转换成标准 RLDS 数据集，方便下游训练和共享。

### 前置条件

- 已使用 `run_simple.py` 或 `main_generator.py` 生成原始数据，且保存在 `./vla_output/` 目录。
- 已创建并安装好 `rlds_env` conda 环境（用于TensorFlow / RLDS依赖）。

示例创建方式（简化版）：

```bash
conda create -n rlds_env python=3.8 -y
conda activate rlds_env
pip install tensorflow tensorflow-datasets numpy pillow
```

### 方法1：使用自动化脚本（推荐）

```bash
cd path/to/vla_path_generate
./convert_rlds.sh
```

或显式使用 bash：

```bash
cd path/to/vla_path_generate
bash convert_rlds.sh
```

脚本会从 `./vla_output` 读取数据，并在 `./vla_rlds` 下写出 RLDS 数据集。

### 方法2：手动运行 Python 转换脚本

如果你希望显式控制环境或参数，可以直接调用 Python 脚本：

```bash
# 激活RLDS环境
conda activate rlds_env

# 进入项目目录
cd path/to/vla_path_generate

# 运行转换脚本
python convert_to_rlds.py \
    --input_dir ./vla_output \
    --output_dir ./vla_rlds

# 完成后可按需退出环境
conda deactivate
```

### 方法3：使用 conda run（无需显式激活环境）

如果不想改变当前 shell 的环境，可以使用 `conda run`：

```bash
cd path/to/vla_path_generate
conda run -n rlds_env python convert_to_rlds.py \
    --input_dir ./vla_output \
    --output_dir ./vla_rlds
```

### 转换输出结构

转换成功后，将在 `./vla_rlds/` 目录下看到：

```text
vla_rlds/
├── vla_franka_manipulation.json      # JSON格式的数据集（用于调试和可视化）
└── tfrecords/                        # TFRecord格式的数据集（用于训练）
    ├── vla_franka_manipulation-00000-of-00001.tfrecord
    └── ...
```

其中：

- **JSON 文件**：包含所有 episode 的元数据与索引信息，方便快速检查。
- **TFRecord 文件**：标准 TensorFlow 数据集格式，可直接在训练脚本中加载。

### 快速检查与验证

转换完成后，可用以下命令做最基本的检查：

```bash
cd path/to/vla_path_generate

ls -lh vla_rlds/
ls -lh vla_rlds/tfrecords/

# 查看 JSON 前100行
head -n 100 vla_rlds/vla_franka_manipulation.json
```

如需在 Python 中加载，可参考：

```python
import json

# 从 JSON 加载（调试用）
with open('vla_rlds/vla_franka_manipulation.json', 'r') as f:
    dataset = json.load(f)
    print(f"Loaded {len(dataset['episodes'])} episodes")

# 从 TFRecord 加载（训练用）
from rlds_writer import load_rlds_dataset

dataset = load_rlds_dataset('vla_rlds/tfrecords/*.tfrecord')
for episode in dataset.take(1):
    print(episode)
```

### RLDS转换常见问题简要汇总

- **找不到 `rlds_env` 环境**：
  - 请先用 `conda create -n rlds_env python=3.8 -y` 创建环境，并安装 `tensorflow` / `tensorflow-datasets` 等依赖。
- **提示找不到输入目录**：
  - 确认 `./vla_output/` 已存在，并包含 `dataset_info.json` 和若干 `episode_xxxx/` 子目录。
- **内存不足**：
  - 可以在 `vla_output/dataset_info.json` 中调小 `shard_size`，减小单个 TFRecord 的 episode 数量。

更多细节与完整说明请参考 `RLDS_CONVERSION_GUIDE.md`。

## 代码模块说明

### FrankaRRTController

Franka机械臂的RRT路径规划控制器。

```python
from franka_rrt_controller import FrankaRRTController

# 初始化
controller = FrankaRRTController(
    robot_articulation=robot,
    robot_description_path="configs/franka_description.yaml",
    urdf_path="configs/franka.urdf",
    rrt_config_path="configs/franka_rrt_config.yaml"
)

# 规划到目标位置
path = controller.plan_to_target_position(
    target_position=np.array([0.5, 0.0, 0.3]),
    target_orientation=np.array([1.0, 0.0, 0.0, 0.0])
)

# 执行路径
trajectory = controller.execute_path(path)
```

### RLDSWriter

RLDS格式数据写入器。

```python
from rlds_writer import RLDSWriter

# 初始化
writer = RLDSWriter(
    dataset_name="my_dataset",
    output_dir="./output"
)

# 创建episode
episode_data = writer.create_episode_from_trajectory(
    images=images,
    actions=actions,
    joint_positions=joint_positions,
    ee_positions=ee_positions,
    ee_orientations=ee_orientations,
    instruction="拿起红色方块"
)

# 添加episode
writer.add_episode(episode_data, episode_id=0)

# 保存数据
writer.save_to_tfrecord()
writer.save_to_json()
```

### VLADataGenerator

主数据生成器。

```python
from vla_data_generator import VLADataGenerator

# 初始化
generator = VLADataGenerator(config, simulation_app)

# 设置场景
generator.setup_scene()

# 设置RRT控制器
generator.setup_rrt_controller(...)

# 生成episode
episode_data = generator.generate_episode(task_config, episode_id)
```

## 常见问题

### Q1: RRT路径规划失败

**A**: 可能的原因和解决方案：

1. **障碍物太多**: 减少场景中的障碍物或调整其位置
2. **目标位置不可达**: 检查目标位置是否在机械臂工作空间内
3. **步长过大**: 减小RRT配置中的 `step_size`
4. **迭代次数不足**: 增加 `max_iterations`

### Q2: 相机无法捕获图像

**A**: 解决方案：

1. 确保 `world.step(render=True)` 设置了 `render=True`
2. 检查相机位置是否正确
3. 增加仿真步数，等待渲染完成

### Q3: URDF文件找不到

**A**: 

1. 确认URDF文件路径正确
2. 可以从Franka官方仓库下载：
   ```bash
   git clone https://github.com/frankaemika/franka_ros.git
   cp franka_ros/franka_description/robots/panda_arm.urdf.xacro configs/
   ```

### Q4: TensorFlow导入错误

**A**: 

```bash
# 安装兼容版本的TensorFlow
pip install tensorflow==2.12.0 tensorflow-datasets==4.9.0
```

### Q5: 夹爪控制问题

**A**: 当前版本提供了简化的夹爪控制。对于更精确的控制，需要：

1. 在URDF中定义夹爪关节
2. 实现物理约束来附着物体
3. 使用Isaac Sim的抓取API

## 性能优化

### 1. GPU内存优化

如果遇到GPU内存不足：

```python
# 在CONFIG中添加
CONFIG = {
    ...
    "carb_settings": {
        "/rtx/memory/poolSize": 1024 * 1024 * 1024,  # 1GB
        "/rtx/raytracing/enable": False,  # 禁用光线追踪
    }
}
```

### 2. 数据生成速度

- 使用 `headless=True` 模式
- 减少仿真步数
- 降低图像分辨率
- 批量生成多个episodes

### 3. 存储优化

- 调整 `shard_size` 来控制每个文件大小
- 使用JPEG压缩图像（在RLDSWriter中已实现）
- 只保存关键帧

## 扩展和定制

### 添加新的任务类型

```python
# 在config.json的tasks中添加
{
  "instruction": "堆叠方块",
  "target_object": "cube1",
  "target_position": [0.3, 0.3, 0.1],
  "custom_params": {
    "stack_on": "cube2",
    "approach_angle": 90
  }
}
```

### 添加新的传感器

```python
# 在VLADataGenerator中
def setup_wrist_camera(self):
    self.wrist_camera = self.Camera(
        prim_path="/World/franka/panda_hand/wrist_camera",
        name="wrist_camera",
        position=np.array([0.0, 0.0, 0.05]),
        resolution=(320, 240)
    )
```

### 自定义动作空间

```python
# 修改_execute_and_record方法中的action计算
# 例如：使用笛卡尔空间增量
action = ee_pos_next - ee_pos_current
```

## 参考资料

- [Isaac Sim官方文档](https://docs.omniverse.nvidia.com/isaacsim/)
- [RRT算法文档](https://docs.isaacsim.omniverse.nvidia.com/py/source/extensions/isaacsim.robot_motion.motion_generation/)
- [RLDS格式规范](https://github.com/google-research/rlds)
- [Franka Panda机械臂](https://www.franka.de/)
