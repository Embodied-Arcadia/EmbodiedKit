"""
Franka机械臂RRT路径规划控制器
使用Isaac Sim的RRT路径规划接口为Franka机械臂生成无碰撞路径
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import os


class FrankaRRTController:
    """Franka机械臂的RRT路径规划控制器"""
    
    def __init__(
        self,
        robot_articulation,
        robot_description_path: str,
        urdf_path: str,
        rrt_config_path: str,
        end_effector_name: str = "panda_hand"
    ):
        """
        初始化Franka RRT控制器
        
        Args:
            robot_articulation: Isaac Sim机器人关节对象
            robot_description_path: 机器人描述文件路径
            urdf_path: URDF文件路径
            rrt_config_path: RRT配置文件路径
            end_effector_name: 末端执行器名称
        """
        self.robot = robot_articulation
        self.end_effector_name = end_effector_name
        
        # 延迟导入Isaac Sim模块
        from isaacsim.robot_motion.motion_generation.lula import RRT
        from isaacsim.robot_motion.motion_generation import LulaCSpaceTrajectoryGenerator
        
        # 初始化RRT规划器
        self.rrt_planner = RRT(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_path,
            end_effector_frame_name=end_effector_name
        )
        
        # RRT参数配置
        self.rrt_planner.set_max_iterations(10000)
        self.rrt_planner.set_random_seed(42)
        
        # 设置步长等参数
        self.rrt_planner.set_param("step_size", 0.1)
        
        # 获取并存储活动关节信息
        self.active_joints = self.rrt_planner.get_active_joints()
        self.num_active_joints = len(self.active_joints)
        
        # Franka Panda关节限位 (弧度)
        # 参考: https://frankaemika.github.io/docs/control_parameters.html
        self.joint_limits = {
            'lower': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            'upper': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        }
        
        # 默认/初始关节位置 (安全的home位置)
        self.default_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # 初始化时间最优轨迹生成器
        try:
            self.trajectory_generator = LulaCSpaceTrajectoryGenerator(
                robot_description_path=robot_description_path,
                urdf_path=urdf_path
            )
            print("✅ Time-optimal trajectory generator initialized")
            self.use_time_optimal_trajectory = True
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize trajectory generator: {e}")
            print("   Falling back to linear interpolation")
            self.trajectory_generator = None
            self.use_time_optimal_trajectory = False
        
        print(f"Franka RRT Controller initialized with end effector: {end_effector_name}")
        print(f"Active joints ({self.num_active_joints}): {self.active_joints}")
        print(f"Joint limits configured")
    
    def get_active_joint_positions(self, all_joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取活动关节的位置（排除夹爪等非活动关节）
        
        Args:
            all_joint_positions: 所有关节位置，如果为None则从机器人获取
            
        Returns:
            活动关节位置数组
        """
        if all_joint_positions is None:
            all_joint_positions = self.robot.get_joint_positions()
        
        # 只返回前num_active_joints个关节（通常Franka臂关节为前7个）
        active_positions = all_joint_positions[:self.num_active_joints]
        
        return active_positions
    
    def validate_joint_positions(self, joint_positions: np.ndarray, tolerance: float = 0.01) -> Tuple[bool, str]:
        """
        验证关节位置是否在限位范围内
        
        Args:
            joint_positions: 关节位置数组
            tolerance: 容差值（弧度）
            
        Returns:
            (是否有效, 错误信息) 元组
        """
        if len(joint_positions) != self.num_active_joints:
            return False, f"Joint positions length mismatch: expected {self.num_active_joints}, got {len(joint_positions)}"
        
        lower_limits = self.joint_limits['lower'] - tolerance
        upper_limits = self.joint_limits['upper'] + tolerance
        
        # 检查每个关节
        violations = []
        for i, (pos, lower, upper) in enumerate(zip(joint_positions, lower_limits, upper_limits)):
            if pos < lower or pos > upper:
                violations.append(
                    f"Joint {i} ({self.active_joints[i]}): {pos:.4f} not in [{lower:.4f}, {upper:.4f}]"
                )
        
        if violations:
            return False, "; ".join(violations)
        
        return True, "Valid"
    
    def clip_joint_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        将关节位置限制在限位范围内
        
        Args:
            joint_positions: 关节位置数组
            
        Returns:
            限制后的关节位置
        """
        clipped = np.clip(
            joint_positions[:self.num_active_joints],
            self.joint_limits['lower'],
            self.joint_limits['upper']
        )
        return clipped
    
    def reset_to_default_position(self):
        """
        将机器人重置到默认位置
        """
        print("Resetting robot to default position...")
        self.set_active_joint_positions(self.default_joint_positions)
        print(f"  Default position: {self.default_joint_positions}")
    
    def generate_time_optimal_trajectory(
        self,
        waypoints: np.ndarray,
        physics_dt: float = 1.0 / 60.0
    ) -> Optional[np.ndarray]:
        """
        使用Lula轨迹生成器生成时间最优的平滑轨迹
        
        Args:
            waypoints: RRT路径点 (N x m)
            physics_dt: 物理时间步长
            
        Returns:
            时间最优的轨迹数组，如果失败返回None
        """
        if not self.use_time_optimal_trajectory or self.trajectory_generator is None:
            return waypoints  # 返回原始路径
        
        try:
            from isaacsim.robot_motion.motion_generation import ArticulationTrajectory
            
            # 使用Lula生成时间最优轨迹
            lula_trajectory = self.trajectory_generator.compute_c_space_trajectory(waypoints)
            
            if lula_trajectory is None:
                print("⚠️  Time-optimal trajectory generation failed, using original path")
                return waypoints
            
            # 将Lula轨迹转换为关节位置序列
            articulation_trajectory = ArticulationTrajectory(
                self.robot,
                lula_trajectory,
                physics_dt
            )
            
            # 获取动作序列
            action_sequence = articulation_trajectory.get_action_sequence()
            
            # 从动作序列提取关节位置
            trajectory_points = []
            for action in action_sequence:
                # 提取活动关节的位置
                joint_positions = np.zeros(self.num_active_joints)
                for idx, joint_idx in enumerate(action.joint_indices[:self.num_active_joints]):
                    if joint_idx < len(action.joint_positions):
                        joint_positions[idx] = action.joint_positions[joint_idx]
                trajectory_points.append(joint_positions)
            
            if len(trajectory_points) == 0:
                print("⚠️  No trajectory points generated, using original path")
                return waypoints
            
            trajectory_array = np.array(trajectory_points)
            print(f"✅ Time-optimal trajectory generated: {len(waypoints)} waypoints → {len(trajectory_array)} smooth steps")
            return trajectory_array
            
        except Exception as e:
            print(f"⚠️  Error generating time-optimal trajectory: {e}")
            print("   Falling back to original path")
            return waypoints
    
    def set_active_joint_positions(self, active_positions: np.ndarray, validate: bool = True):
        """
        设置活动关节位置（保持夹爪等非活动关节不变）
        
        Args:
            active_positions: 活动关节位置数组
            validate: 是否验证关节限位
        """
        # 验证关节限位
        if validate:
            is_valid, msg = self.validate_joint_positions(active_positions)
            if not is_valid:
                print(f"Warning: Invalid joint positions - {msg}")
                print("  Clipping to joint limits...")
                active_positions = self.clip_joint_positions(active_positions)
        
        # 获取当前所有关节位置
        all_joint_positions = self.robot.get_joint_positions()
        
        # 更新活动关节位置，保持其他关节不变
        all_joint_positions[:self.num_active_joints] = active_positions[:self.num_active_joints]
        
        # 设置所有关节位置
        self.robot.set_joint_positions(all_joint_positions)
    
    def set_robot_base_pose(self, position: np.ndarray, orientation: np.ndarray):
        """
        设置机器人基座位姿
        
        Args:
            position: 3x1位置向量
            orientation: 4x1四元数(x,y,z,w)
        """
        self.rrt_planner.set_robot_base_pose(position, orientation)
        print(f"Set robot base pose: position={position}, orientation={orientation}")
    
    def add_obstacles(self, obstacles: List[Dict]):
        """
        添加障碍物到RRT规划器
        
        Args:
            obstacles: 障碍物列表，每个障碍物包含类型和参数
        """
        try:
            from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, FixedCuboid
        except ImportError:
            from isaacsim.core.api.objects.cuboid import DynamicCuboid, FixedCuboid
            from isaacsim.core.api.objects.sphere import DynamicSphere
        
        for obs_info in obstacles:
            obs_type = obs_info.get("type", "cuboid")
            position = np.array(obs_info.get("position", [0, 0, 0]))
            
            if obs_type == "cuboid":
                size = obs_info.get("size", [0.1, 0.1, 0.1])
                obstacle = FixedCuboid(
                    prim_path=obs_info.get("prim_path", "/World/obstacle"),
                    position=position,
                    size=np.array(size)
                )
                self.rrt_planner.add_cuboid(obstacle, static=True)
            elif obs_type == "sphere":
                radius = obs_info.get("radius", 0.05)
                obstacle = DynamicSphere(
                    prim_path=obs_info.get("prim_path", "/World/obstacle"),
                    position=position,
                    radius=radius
                )
                self.rrt_planner.add_sphere(obstacle, static=True)
            
            print(f"Added {obs_type} obstacle at {position}")
    
    def plan_to_target_position(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        current_joint_positions: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        规划到目标位置的路径
        
        Args:
            target_position: 目标位置(3x1)
            target_orientation: 目标方向四元数(4x1), 如果为None则仅考虑位置
            current_joint_positions: 当前关节位置，如果为None则从机器人获取
            
        Returns:
            路径数组 (N x m)，N为路径点数，m为活动关节数；如果规划失败返回None
        """
        # 获取当前关节位置（只获取活动关节）
        if current_joint_positions is None:
            current_joint_positions = self.get_active_joint_positions()
        else:
            # 如果提供的关节位置包含所有关节，只取活动关节
            if len(current_joint_positions) > self.num_active_joints:
                current_joint_positions = current_joint_positions[:self.num_active_joints]
        
        # 验证当前关节位置
        is_valid, msg = self.validate_joint_positions(current_joint_positions)
        if not is_valid:
            print(f"⚠️  Warning: Current joint positions are invalid!")
            print(f"   Reason: {msg}")
            print(f"   Attempting to clip to valid range...")
            current_joint_positions = self.clip_joint_positions(current_joint_positions)
            # 更新机器人位置到有效状态
            self.set_active_joint_positions(current_joint_positions, validate=False)
            # 再次验证
            is_valid, msg = self.validate_joint_positions(current_joint_positions)
            if not is_valid:
                print(f"❌ Error: Still invalid after clipping. Resetting to default position.")
                self.reset_to_default_position()
                current_joint_positions = self.default_joint_positions
        
        print(f"Current active joint positions ({len(current_joint_positions)}): {current_joint_positions}")
        
        # 设置末端执行器目标
        if target_orientation is not None:
            self.rrt_planner.set_end_effector_target(
                target_translation=target_position,
                target_orientation=target_orientation
            )
            print(f"Planning to target pose: pos={target_position}, ori={target_orientation}")
        else:
            self.rrt_planner.set_end_effector_target(
                target_translation=target_position
            )
            print(f"Planning to target position: {target_position}")
        
        # 计算路径（watched_joint_positions为空数组，因为Franka不需要watched joints）
        watched_joints = []
        path = self.rrt_planner.compute_path(
            active_joint_positions=current_joint_positions,
            watched_joint_positions=np.array(watched_joints)
        )
        
        if path is not None:
            print(f"Path planning successful! Path length: {len(path)} waypoints")
        else:
            print("Path planning failed!")
        
        return path
    
    def plan_to_joint_configuration(
        self,
        target_joint_positions: np.ndarray,
        current_joint_positions: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        规划到目标关节配置的路径
        
        Args:
            target_joint_positions: 目标关节位置
            current_joint_positions: 当前关节位置，如果为None则从机器人获取
            
        Returns:
            路径数组 (N x m)；如果规划失败返回None
        """
        # 获取当前关节位置（只获取活动关节）
        if current_joint_positions is None:
            current_joint_positions = self.get_active_joint_positions()
        else:
            # 如果提供的关节位置包含所有关节，只取活动关节
            if len(current_joint_positions) > self.num_active_joints:
                current_joint_positions = current_joint_positions[:self.num_active_joints]
        
        # 验证当前关节位置
        is_valid, msg = self.validate_joint_positions(current_joint_positions)
        if not is_valid:
            print(f"⚠️  Warning: Current joint positions are invalid!")
            print(f"   Reason: {msg}")
            current_joint_positions = self.clip_joint_positions(current_joint_positions)
            self.set_active_joint_positions(current_joint_positions, validate=False)
        
        # 确保目标关节位置也是活动关节
        if len(target_joint_positions) > self.num_active_joints:
            target_joint_positions = target_joint_positions[:self.num_active_joints]
        
        # 设置关节空间目标
        self.rrt_planner.set_cspace_target(target_joint_positions)
        print(f"Planning to target joint configuration: {target_joint_positions}")
        
        # 计算路径
        watched_joints = []
        path = self.rrt_planner.compute_path(
            active_joint_positions=current_joint_positions,
            watched_joint_positions=np.array(watched_joints)
        )
        
        if path is not None:
            print(f"Path planning successful! Path length: {len(path)} waypoints")
        else:
            print("Path planning failed!")
        
        return path
    
    def execute_path(
        self,
        path: np.ndarray,
        speed_factor: float = 1.0
    ) -> List[Dict]:
        """
        执行规划的路径并记录轨迹数据
        
        Args:
            path: 路径数组 (N x m)
            speed_factor: 速度因子，控制执行速度
            
        Returns:
            轨迹数据列表，每个元素包含关节位置、末端执行器位姿等信息
        """
        trajectory = []
        
        for i, waypoint in enumerate(path):
            # 设置活动关节目标位置
            self.set_active_joint_positions(waypoint)
            
            # 获取当前末端执行器位姿
            ee_position, ee_orientation = self.get_end_effector_pose()
            
            # 记录轨迹点
            trajectory_point = {
                "step": i,
                "joint_positions": waypoint.tolist(),
                "ee_position": ee_position.tolist(),
                "ee_orientation": ee_orientation.tolist()
            }
            trajectory.append(trajectory_point)
            
            if i % 10 == 0:
                print(f"Executing waypoint {i}/{len(path)}")
        
        print(f"Path execution complete. Trajectory length: {len(trajectory)}")
        return trajectory
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器当前位姿
        
        Returns:
            (position, orientation) 元组
        """
        # 通过RRT规划器的正运动学获取末端执行器位姿
        current_joints = self.get_active_joint_positions()
        ee_translation, ee_rotation_matrix = self.rrt_planner.get_end_effector_pose(
            active_joint_positions=current_joints,
            frame_name=self.end_effector_name
        )
        
        # 将旋转矩阵转换为四元数
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_matrix(ee_rotation_matrix)
        ee_orientation = rotation.as_quat()  # [x, y, z, w]
        
        return ee_translation, ee_orientation
    
    def get_active_joints(self) -> List[str]:
        """获取活动关节列表"""
        return self.rrt_planner.get_active_joints()
    
    def reset(self):
        """重置RRT规划器状态"""
        self.rrt_planner.reset()
        print("RRT planner reset")


def create_default_franka_config(config_dir: str) -> Tuple[str, str, str]:
    """
    创建默认的Franka配置文件
    
    Args:
        config_dir: 配置文件保存目录
        
    Returns:
        (robot_description_path, urdf_path, rrt_config_path) 元组
    """
    # 使用已有的完整配置文件
    robot_desc_path = os.path.join(config_dir, "rmpflow/robot_descriptor.yaml")
    urdf_path = os.path.join(config_dir, "lula_franka_gen.urdf")
    rrt_config_path = os.path.join(config_dir, "rrt/franka_rrt_config.yaml")
    
    # 检查配置文件是否存在
    if not os.path.exists(robot_desc_path):
        print(f"Warning: Robot descriptor not found at {robot_desc_path}")
        print("Please ensure Franka configuration files are properly installed")
    
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file not found at {urdf_path}")
        print("Please ensure Franka URDF is properly installed")
    
    if not os.path.exists(rrt_config_path):
        print(f"Warning: RRT config not found at {rrt_config_path}")
        print("Please ensure RRT configuration is properly created")
    else:
        print(f"Using existing Franka configuration files in {config_dir}")
    
    return robot_desc_path, urdf_path, rrt_config_path

