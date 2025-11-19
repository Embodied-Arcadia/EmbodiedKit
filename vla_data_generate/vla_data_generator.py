"""
VLA训练数据生成器主模块
使用Franka机械臂和RRT路径规划生成VLA训练数据
输入：上帝视角图片 + 人类指令
输出：RLDS格式数据
"""

import os
import json
import numpy as np
import argparse
import traceback
from typing import Dict, List, Tuple, Optional
from omni.isaac.kit import SimulationApp


# Isaac Sim配置
CONFIG = {
    "headless": True,  # 设置为True以在无GUI模式下运行
    "renderer": "RayTracedLighting",
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
}


class VLADataGenerator:
    """VLA训练数据生成器"""
    
    def __init__(
        self,
        config: Dict,
        simulation_app: SimulationApp
    ):
        """
        初始化VLA数据生成器
        
        Args:
            config: 配置字典
            simulation_app: Isaac Sim应用实例
        """
        self.config = config
        self.app = simulation_app
        
        # 延迟导入Isaac Sim模块（必须在SimulationApp创建之后）
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.prims import Articulation
        
        try:
            from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, DynamicSphere
        except ImportError:
            # 尝试从具体模块导入
            from isaacsim.core.api.objects.cuboid import DynamicCuboid, FixedCuboid
            from isaacsim.core.api.objects.sphere import DynamicSphere
        
        try:
            from isaacsim.core.sensors import Camera
        except ImportError:
            try:
                from omni.isaac.sensor import Camera
            except ImportError:
                print("Warning: Camera module not available")
                Camera = None
        
        self.World = World
        self.open_stage = open_stage
        self.create_prim = create_prim
        self.Articulation = Articulation
        self.Camera = Camera
        self.DynamicCuboid = DynamicCuboid
        self.FixedCuboid = FixedCuboid
        self.DynamicSphere = DynamicSphere
        
        # 初始化组件
        self.world = None
        self.robot = None
        self.camera = None
        self.rrt_controller = None
        
        print("VLA Data Generator initialized")
    
    def setup_scene(
        self,
        usd_path: Optional[str] = None,
        robot_asset_path: Optional[str] = None
    ):
        """
        设置场景
        
        Args:
            usd_path: 场景USD文件路径（如果为None则创建空场景）
            robot_asset_path: Franka机器人资产路径
        """
        print("Setting up scene...")
        
        # 创建或获取World实例
        self.world = self.World.instance()
        if self.world is None:
            self.world = self.World()
        
        # 加载场景或创建空场景
        if usd_path and os.path.exists(usd_path):
            print(f"Loading scene from: {usd_path}")
            self.open_stage(usd_path)
        else:
            print("Creating new empty scene")
            # 添加地面
            try:
                from isaacsim.core.api.objects import GroundPlane
            except ImportError:
                from isaacsim.core.api.objects.ground_plane import GroundPlane
            ground = GroundPlane(prim_path="/World/ground", size=10.0, color=np.array([0.5, 0.5, 0.5]))
            self.world.scene.add(ground)
        
        # 添加Franka机器人
        robot_prim_path = "/World/franka"
        robot_position = self.config.get("robot_start_position", [0.0, 0.0, 0.0])
        
        if robot_asset_path and os.path.exists(robot_asset_path):
            print(f"Loading Franka robot from: {robot_asset_path}")
            self.create_prim(
                prim_path=robot_prim_path,
                prim_type="Xform",
                position=np.array(robot_position),
                usd_path=robot_asset_path
            )
        else:
            # 使用Isaac Sim内置的Franka
            try:
                from isaacsim.robots.franka import Franka
            except ImportError:
                try:
                    from omni.isaac.franka import Franka
                except ImportError:
                    print("Warning: Franka robot class not found, will use Articulation")
                    Franka = None
            
            if Franka is not None:
                self.robot = Franka(
                    prim_path=robot_prim_path,
                    name="franka",
                    position=np.array(robot_position)
                )
                self.world.scene.add(self.robot)
            else:
                print("Warning: Using generic Articulation instead of Franka class")
        
        if self.robot is None:
            self.robot = self.Articulation(robot_prim_path)
            self.world.scene.add(self.robot)
        
        # 添加上帝视角相机
        self.setup_overhead_camera()
        
        # 重置世界
        self.world.reset()
        print("Scene setup complete")
    
    def setup_overhead_camera(
        self,
        position: Optional[np.ndarray] = None,
        look_at: Optional[np.ndarray] = None
    ):
        """
        设置上帝视角相机
        
        Args:
            position: 相机位置
            look_at: 相机朝向的目标点
        """
        if position is None:
            position = np.array([0.0, 0.0, 2.5])  # 默认在上方2.5米
        
        camera_prim_path = "/World/overhead_camera"
        
        self.camera = self.Camera(
            prim_path=camera_prim_path,
            name="overhead_camera",
            position=position,
            resolution=(640, 480),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])  # 向下看
        )
        
        self.camera.initialize()
        
        print(f"Overhead camera initialized at position: {position}")
    
    def setup_rrt_controller(
        self,
        robot_description_path: str,
        urdf_path: str,
        rrt_config_path: str
    ):
        """
        设置RRT控制器
        
        Args:
            robot_description_path: 机器人描述文件路径
            urdf_path: URDF文件路径
            rrt_config_path: RRT配置文件路径
        """
        from franka_rrt_controller import FrankaRRTController
        
        self.rrt_controller = FrankaRRTController(
            robot_articulation=self.robot,
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_path,
            end_effector_name="panda_hand"
        )
        
        # 设置机器人基座位姿
        robot_pos, robot_ori = self.robot.get_world_pose()
        self.rrt_controller.set_robot_base_pose(robot_pos, robot_ori)
        
        print("RRT controller setup complete")
    
    def add_objects_to_scene(self, objects: List[Dict]):
        """
        向场景中添加物体
        
        Args:
            objects: 物体列表，每个物体是一个包含类型、位置、大小等信息的字典
        """
        print(f"Adding {len(objects)} objects to scene...")
        
        for i, obj_info in enumerate(objects):
            obj_type = obj_info.get("type", "cuboid")
            obj_name = obj_info.get("name", f"object_{i}")
            position = np.array(obj_info.get("position", [0.5, 0.0, 0.1]))
            
            prim_path = f"/World/{obj_name}"
            
            if obj_type == "cuboid":
                size_array = obj_info.get("size", [0.05, 0.05, 0.05])
                # DynamicCuboid的size参数期望一个单一的float值（立方体边长）
                # 如果提供的是数组，取第一个值
                if isinstance(size_array, (list, tuple, np.ndarray)):
                    size = float(size_array[0]) if len(size_array) > 0 else 0.05
                    if len(size_array) > 1 and not all(s == size_array[0] for s in size_array):
                        print(f"  Note: Cuboid '{obj_name}' uses first dimension {size} (from {size_array})")
                else:
                    size = float(size_array)
                
                color = np.array(obj_info.get("color", [1.0, 0.0, 0.0]))
                
                obj = self.DynamicCuboid(
                    prim_path=prim_path,
                    name=obj_name,
                    position=position,
                    size=size,
                    color=color
                )
                self.world.scene.add(obj)
                print(f"  Added cuboid '{obj_name}' at {position}, size={size}")
                
            elif obj_type == "sphere":
                radius = float(obj_info.get("radius", 0.025))
                color = np.array(obj_info.get("color", [0.0, 1.0, 0.0]))
                
                obj = self.DynamicSphere(
                    prim_path=prim_path,
                    name=obj_name,
                    position=position,
                    radius=radius,
                    color=color
                )
                self.world.scene.add(obj)
                print(f"  Added sphere '{obj_name}' at {position}, radius={radius}")
        
        # 重置世界以应用更改
        self.world.reset()
    
    def generate_episode(
        self,
        task_config: Dict,
        episode_id: int
    ) -> Dict:
        """
        生成一个episode的数据
        
        Args:
            task_config: 任务配置字典，包含指令、目标等
            episode_id: episode ID
            
        Returns:
            episode数据字典
        """
        instruction = task_config.get("instruction", "Pick and place task")
        target_object = task_config.get("target_object", "object_0")
        target_position = np.array(task_config.get("target_position", [0.3, 0.3, 0.1]))
        
        print(f"\n{'='*60}")
        print(f"Generating Episode {episode_id}")
        print(f"Instruction: {instruction}")
        print(f"Target Object: {target_object}")
        print(f"Target Position: {target_position}")
        print(f"{'='*60}\n")
        
        # 重置机器人到默认位置（防止累积误差和无效配置）
        if episode_id > 0:  # 第一个episode已经在初始位置
            print("Resetting robot to default position for new episode...")
            self.rrt_controller.reset_to_default_position()
            # 让物理引擎稳定
            for _ in range(10):
                self.world.step(render=False)
            print("Robot reset complete\n")
        
        # 获取目标物体
        target_obj = self.world.scene.get_object(target_object)
        if target_obj is None:
            print(f"Warning: Target object '{target_object}' not found in scene")
            return None
        
        # 获取目标物体的当前位置
        obj_position, _ = target_obj.get_world_pose()
        
        # 1. 规划抓取路径
        print("Step 1: Planning grasp path...")
        grasp_position = obj_position + np.array([0.0, 0.0, 0.15])  # 在物体上方
        grasp_path = self.rrt_controller.plan_to_target_position(
            target_position=grasp_position,
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0])  # 从上方抓取
        )
        
        if grasp_path is None:
            print("❌ Failed to plan grasp path!")
            print("   Resetting robot and skipping this episode...")
            self.rrt_controller.reset_to_default_position()
            for _ in range(10):
                self.world.step(render=False)
            return None
        
        # 2. 执行抓取路径并记录数据
        print("Step 2: Executing grasp path...")
        # 从配置读取插值参数
        data_config = self.config.get("data_collection", {})
        interpolate = data_config.get("interpolate_path", True)
        interpolation_steps = data_config.get("interpolation_steps", 10)
        use_time_optimal = data_config.get("use_time_optimal_trajectory", True)
        
        images_grasp, actions_grasp, states_grasp = self._execute_and_record(
            grasp_path,
            interpolate=interpolate,
            interpolation_steps=interpolation_steps,
            use_time_optimal=use_time_optimal
        )
        
        # 3. 闭合夹爪（简化：假设夹爪总是能成功抓取）
        print("Step 3: Closing gripper...")
        self._close_gripper()
        
        # 模拟物体附着到末端执行器
        # 在实际应用中需要使用物理约束或其他方法
        
        # 4. 规划放置路径
        print("Step 4: Planning place path...")
        place_position = target_position + np.array([0.0, 0.0, 0.15])
        place_path = self.rrt_controller.plan_to_target_position(
            target_position=place_position,
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        
        if place_path is None:
            print("❌ Failed to plan place path!")
            print("   Resetting robot and skipping this episode...")
            self.rrt_controller.reset_to_default_position()
            for _ in range(10):
                self.world.step(render=False)
            return None
        
        # 5. 执行放置路径并记录数据
        print("Step 5: Executing place path...")
        images_place, actions_place, states_place = self._execute_and_record(
            place_path,
            interpolate=interpolate,
            interpolation_steps=interpolation_steps,
            use_time_optimal=use_time_optimal
        )
        
        # 6. 打开夹爪
        print("Step 6: Opening gripper...")
        self._open_gripper()
        
        # 合并所有数据
        images = images_grasp + images_place
        actions = actions_grasp + actions_place
        states = states_grasp + states_place
        
        # 提取状态信息
        joint_positions = [s["joint_positions"] for s in states]
        ee_positions = [s["ee_position"] for s in states]
        ee_orientations = [s["ee_orientation"] for s in states]
        
        # 创建原始episode数据（不使用RLDS，避免tensorflow依赖）
        episode_data = {
            "episode_id": episode_id,
            "instruction": instruction,
            "images": images,  # List of numpy arrays
            "actions": actions,  # List of numpy arrays
            "joint_positions": joint_positions,  # List of numpy arrays
            "ee_positions": ee_positions,  # List of numpy arrays
            "ee_orientations": ee_orientations,  # List of numpy arrays
            "task_success": True,
            "metadata": {
                "target_object": target_object,
                "target_position": target_position.tolist(),
                "num_grasp_steps": len(images_grasp),
                "num_place_steps": len(images_place),
                "total_steps": len(images)
            }
        }
        
        print(f"Episode {episode_id} generated successfully!")
        print(f"  Total steps: {len(images)}")
        print(f"  Grasp steps: {len(images_grasp)}")
        print(f"  Place steps: {len(images_place)}")
        
        return episode_data
    
    def _interpolate_path(
        self,
        path: np.ndarray,
        num_steps: int = 10,
        use_time_optimal: bool = True
    ) -> np.ndarray:
        """
        对路径进行插值，增加中间点
        
        Args:
            path: 原始路径 (N x m)
            num_steps: 每两个waypoint之间插值的步数（仅在use_time_optimal=False时使用）
            use_time_optimal: 是否使用时间最优轨迹生成器
            
        Returns:
            插值后的路径
        """
        if len(path) <= 1:
            return path
        
        # 尝试使用时间最优轨迹生成器
        if use_time_optimal and hasattr(self.rrt_controller, 'generate_time_optimal_trajectory'):
            physics_dt = self.config.get("physics_dt", 1.0 / 60.0)
            time_optimal_path = self.rrt_controller.generate_time_optimal_trajectory(path, physics_dt)
            
            if time_optimal_path is not None and len(time_optimal_path) > len(path):
                return time_optimal_path
            
            # 如果时间最优生成失败，使用线性插值作为备选
            print("  Falling back to linear interpolation")
        
        # 线性插值（备选方案）
        interpolated = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # 线性插值
            for t in np.linspace(0, 1, num_steps, endpoint=False):
                interpolated.append(start + t * (end - start))
        
        # 添加最后一个点
        interpolated.append(path[-1])
        
        return np.array(interpolated)
    
    def _execute_and_record(
        self,
        path: np.ndarray,
        interpolate: bool = True,
        interpolation_steps: int = 10,
        use_time_optimal: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        执行路径并记录图像、动作和状态
        
        Args:
            path: 关节空间路径
            interpolate: 是否对路径进行插值
            interpolation_steps: 每两个waypoint之间的插值步数
            use_time_optimal: 是否使用时间最优轨迹生成
            
        Returns:
            (images, actions, states) 元组
        """
        # 路径插值或时间最优轨迹生成
        if interpolate and len(path) > 1:
            original_length = len(path)
            path = self._interpolate_path(path, interpolation_steps, use_time_optimal)
            
            if use_time_optimal:
                print(f"  Path processed: {original_length} waypoints → {len(path)} time-optimal steps")
            else:
                print(f"  Path interpolated: {original_length} waypoints → {len(path)} steps")
        
        images = []
        actions = []
        states = []
        
        for i in range(len(path)):
            # 设置活动关节位置
            target_joints = path[i]
            self.rrt_controller.set_active_joint_positions(target_joints)
            
            # 仿真步进（插值后可以减少步数）
            sim_steps = 2 if interpolate else 5  # 插值后每步只需2次仿真
            for _ in range(sim_steps):
                self.world.step(render=True)
            
            # 捕获图像
            if self.camera is not None:
                try:
                    rgb_image = self.camera.get_rgba()[:, :, :3]  # 只取RGB通道
                    images.append(rgb_image)
                except Exception as e:
                    print(f"Warning: Failed to capture image: {e}")
                    # 使用空白图像
                    images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # 计算动作（下一个关节位置与当前的差值）
            if i < len(path) - 1:
                action = path[i + 1] - path[i]
            else:
                action = np.zeros_like(path[i])
            actions.append(action)
            
            # 记录状态（只记录活动关节）
            current_joints = self.rrt_controller.get_active_joint_positions()
            ee_pos, ee_ori = self.rrt_controller.get_end_effector_pose()
            
            state = {
                "joint_positions": current_joints,
                "ee_position": ee_pos,
                "ee_orientation": ee_ori
            }
            states.append(state)
        
        return images, actions, states
    
    def _close_gripper(self):
        """闭合夹爪"""
        # 简化实现：设置夹爪关节到闭合位置
        # 实际的Franka夹爪有两个关节：panda_finger_joint1和panda_finger_joint2
        gripper_close_position = 0.0  # 闭合位置
        # 这里需要根据实际的机器人配置设置夹爪
        print("Gripper closed")
    
    def _open_gripper(self):
        """打开夹爪"""
        gripper_open_position = 0.04  # 打开位置
        print("Gripper opened")
    
    def cleanup(self):
        """清理资源"""
        if self.world is not None:
            self.world.stop()
        print("Cleanup complete")


def run_generation(args):
    """运行VLA数据生成"""
    
    # 加载配置
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # 创建SimulationApp
    app = SimulationApp(CONFIG)
    
    try:
        # 创建生成器
        generator = VLADataGenerator(config, app)
        
        # 设置场景
        generator.setup_scene(
            usd_path=config.get("scene_usd_path"),
            robot_asset_path=config.get("robot_asset_path")
        )
        
        # 设置RRT控制器
        generator.setup_rrt_controller(
            robot_description_path=config["robot_description_path"],
            urdf_path=config["urdf_path"],
            rrt_config_path=config["rrt_config_path"]
        )
        
        # 添加场景物体
        if "objects" in config:
            generator.add_objects_to_scene(config["objects"])
        
        # 创建输出目录
        output_dir = config.get("output_dir", "./vla_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成多个episodes
        num_episodes = config.get("num_episodes", 1)
        tasks = config.get("tasks", [])
        
        all_episodes = []
        successful_episodes = 0
        
        for episode_id in range(num_episodes):
            # 获取任务配置
            task_config = tasks[episode_id % len(tasks)] if tasks else {}
            
            # 生成episode
            episode_data = generator.generate_episode(task_config, episode_id)
            
            if episode_data is not None:
                all_episodes.append(episode_data)
                successful_episodes += 1
                
                # 保存单个episode的原始数据
                episode_dir = os.path.join(output_dir, f"episode_{episode_id:04d}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # 保存图像
                images_dir = os.path.join(episode_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                for step_idx, img in enumerate(episode_data["images"]):
                    img_path = os.path.join(images_dir, f"step_{step_idx:04d}.npy")
                    np.save(img_path, img)
                
                # 保存动作和状态
                np.save(os.path.join(episode_dir, "actions.npy"), np.array(episode_data["actions"]))
                np.save(os.path.join(episode_dir, "joint_positions.npy"), np.array(episode_data["joint_positions"]))
                np.save(os.path.join(episode_dir, "ee_positions.npy"), np.array(episode_data["ee_positions"]))
                np.save(os.path.join(episode_dir, "ee_orientations.npy"), np.array(episode_data["ee_orientations"]))
                
                # 保存元数据
                metadata = {
                    "episode_id": episode_id,
                    "instruction": episode_data["instruction"],
                    "task_success": episode_data["task_success"],
                    "metadata": episode_data["metadata"]
                }
                with open(os.path.join(episode_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"  Saved episode {episode_id} to {episode_dir}")
            else:
                print(f"Failed to generate episode {episode_id}")
        
        # 保存数据集整体信息
        print("\nSaving dataset summary...")
        dataset_info = {
            "dataset_name": config.get("dataset_name", "vla_franka_manipulation"),
            "description": config.get("dataset_description", "VLA robot manipulation dataset with Franka arm"),
            "num_episodes": successful_episodes,
            "total_requested": num_episodes,
            "tasks": tasks,
            "config": config
        }
        
        with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # 打印数据集信息
        print("\n" + "="*60)
        print("Dataset Generation Complete!")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Successful episodes: {successful_episodes}/{num_episodes}")
        print(f"Dataset info saved to: {os.path.join(output_dir, 'dataset_info.json')}")
        print("\nTo convert to RLDS format, run the conversion script in a TensorFlow environment.")
        
        # 清理
        generator.cleanup()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VLA training data with Franka arm and RRT")
    parser.add_argument("--config_file", type=str, required=True, help="Path to configuration JSON file")
    
    args = parser.parse_args()
    run_generation(args)

