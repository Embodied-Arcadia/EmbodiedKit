#!/usr/bin/env python3
"""
VLA 轨迹可视化工具

功能:
1. 3D轨迹可视化（末端执行器轨迹）
2. 关节角度变化曲线
3. 速度和加速度分析
4. 任务执行统计

使用方法:
    python visualize_trajectory.py --data-dir ./vla_output
    python visualize_trajectory.py --episode-file ./vla_output/episode_000.json
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
import tensorflow as tf


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, data_dir: Optional[str] = None, episode_file: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录（包含多个episode）
            episode_file: 单个episode JSON文件路径
        """
        self.data_dir = data_dir
        self.episode_file = episode_file
        self.episodes = []
        
        # 加载数据
        if episode_file:
            self.load_single_episode(episode_file)
        elif data_dir:
            self.load_episodes_from_dir(data_dir)
        else:
            raise ValueError("Must provide either data_dir or episode_file")
    
    def load_single_episode(self, filepath: str):
        """加载单个episode JSON文件"""
        print(f"Loading episode from: {filepath}")
        with open(filepath, 'r') as f:
            episode = json.load(f)
        self.episodes.append(episode)
        print(f"Loaded 1 episode with {len(episode.get('steps', []))} steps")
    
    def load_episodes_from_dir(self, data_dir: str):
        """从目录加载所有episode JSON文件"""
        print(f"Loading episodes from directory: {data_dir}")
        
        # 查找所有JSON文件
        json_files = glob.glob(os.path.join(data_dir, "episode_*.json"))
        
        if not json_files:
            # 尝试从metadata.json读取
            metadata_path = os.path.join(data_dir, "metadata.json")
            if os.path.exists(metadata_path):
                print(f"Loading from metadata: {metadata_path}")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # metadata可能包含episodes列表
                    if 'episodes' in metadata:
                        self.episodes = metadata['episodes']
            else:
                print("Warning: No episode JSON files found")
                return
        else:
            # 加载所有episode文件
            for json_file in sorted(json_files):
                try:
                    with open(json_file, 'r') as f:
                        episode = json.load(f)
                    self.episodes.append(episode)
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
        
        print(f"Loaded {len(self.episodes)} episodes")
    
    def extract_trajectory_data(self, episode: Dict) -> Dict:
        """
        从episode中提取轨迹数据
        
        Returns:
            包含轨迹数据的字典
        """
        steps = episode.get('steps', [])
        
        ee_positions = []
        ee_orientations = []
        joint_positions = []
        timestamps = []
        
        for i, step in enumerate(steps):
            obs = step.get('observation', {})
            
            # 提取末端执行器位置和姿态
            if 'ee_pos' in obs:
                ee_positions.append(obs['ee_pos'])
            if 'ee_ori' in obs:
                ee_orientations.append(obs['ee_ori'])
            
            # 提取关节位置
            if 'state' in obs:
                joint_positions.append(obs['state'])
            
            timestamps.append(i)
        
        return {
            'ee_positions': np.array(ee_positions) if ee_positions else None,
            'ee_orientations': np.array(ee_orientations) if ee_orientations else None,
            'joint_positions': np.array(joint_positions) if joint_positions else None,
            'timestamps': np.array(timestamps),
            'instruction': episode.get('language_instruction', 'N/A'),
            'metadata': episode.get('metadata', {})
        }
    
    def plot_3d_trajectory(self, trajectory_data: Dict, ax: Optional[plt.Axes] = None, 
                          title: str = "End-Effector 3D Trajectory"):
        """
        绘制3D末端执行器轨迹
        
        Args:
            trajectory_data: 轨迹数据
            ax: matplotlib 3D轴（如果为None则创建新的）
            title: 图表标题
        """
        ee_positions = trajectory_data['ee_positions']
        
        if ee_positions is None or len(ee_positions) == 0:
            print("Warning: No end-effector position data to plot")
            return
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # 提取x, y, z坐标
        x = ee_positions[:, 0]
        y = ee_positions[:, 1]
        z = ee_positions[:, 2]
        
        # 绘制轨迹（用颜色表示时间进度）
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        # 时间颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(x) - 1))
        
        # 3D线段绘制（简化版）
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', 
                  label='Start', edgecolors='black', linewidths=2)
        ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', 
                  label='End', edgecolors='black', linewidths=2)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置相同的刻度比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return ax
    
    def plot_joint_trajectories(self, trajectory_data: Dict, ax: Optional[plt.Axes] = None):
        """
        绘制关节角度随时间变化
        
        Args:
            trajectory_data: 轨迹数据
            ax: matplotlib轴
        """
        joint_positions = trajectory_data['joint_positions']
        timestamps = trajectory_data['timestamps']
        
        if joint_positions is None or len(joint_positions) == 0:
            print("Warning: No joint position data to plot")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        num_joints = joint_positions.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
        
        for i in range(num_joints):
            ax.plot(timestamps, joint_positions[:, i], 
                   label=f'Joint {i+1}', color=colors[i], linewidth=2)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax.set_title('Joint Trajectories', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_velocity_acceleration(self, trajectory_data: Dict):
        """
        绘制速度和加速度曲线
        
        Args:
            trajectory_data: 轨迹数据
        """
        ee_positions = trajectory_data['ee_positions']
        
        if ee_positions is None or len(ee_positions) < 3:
            print("Warning: Not enough data to compute velocity/acceleration")
            return
        
        # 计算速度（位置差分）
        velocity = np.diff(ee_positions, axis=0)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # 计算加速度（速度差分）
        acceleration = np.diff(velocity, axis=0)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 速度图
        time_vel = np.arange(len(velocity_magnitude))
        ax1.plot(time_vel, velocity_magnitude, linewidth=2, color='blue')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Velocity (m/step)', fontsize=12)
        ax1.set_title('End-Effector Velocity', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(time_vel, 0, velocity_magnitude, alpha=0.3, color='blue')
        
        # 加速度图
        time_acc = np.arange(len(acceleration_magnitude))
        ax2.plot(time_acc, acceleration_magnitude, linewidth=2, color='red')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Acceleration (m/step²)', fontsize=12)
        ax2.set_title('End-Effector Acceleration', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(time_acc, 0, acceleration_magnitude, alpha=0.3, color='red')
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive_analysis(self, episode_idx: int = 0, save_path: Optional[str] = None):
        """
        生成综合分析图
        
        Args:
            episode_idx: episode索引
            save_path: 保存路径（如果为None则显示）
        """
        if episode_idx >= len(self.episodes):
            print(f"Error: Episode {episode_idx} not found")
            return
        
        episode = self.episodes[episode_idx]
        trajectory_data = self.extract_trajectory_data(episode)
        
        # 创建综合图表
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 3D轨迹
        ax1 = fig.add_subplot(gs[0, :], projection='3d')
        self.plot_3d_trajectory(trajectory_data, ax=ax1)
        
        # 2. 关节角度
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_joint_trajectories(trajectory_data, ax=ax2)
        
        # 3. XYZ位置分量
        ax3 = fig.add_subplot(gs[2, 0])
        if trajectory_data['ee_positions'] is not None:
            ee_pos = trajectory_data['ee_positions']
            time = trajectory_data['timestamps']
            ax3.plot(time, ee_pos[:, 0], label='X', linewidth=2)
            ax3.plot(time, ee_pos[:, 1], label='Y', linewidth=2)
            ax3.plot(time, ee_pos[:, 2], label='Z', linewidth=2)
            ax3.set_xlabel('Time Step', fontsize=12)
            ax3.set_ylabel('Position (m)', fontsize=12)
            ax3.set_title('End-Effector XYZ Positions', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. 任务信息统计
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        # 统计信息
        metadata = trajectory_data['metadata']
        instruction = trajectory_data['instruction']
        num_steps = len(trajectory_data['timestamps'])
        
        stats_text = f"""
        Task Information:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Instruction: {instruction}
        
        Episode Statistics:
        • Total Steps: {num_steps}
        • Grasp Steps: {metadata.get('num_grasp_steps', 'N/A')}
        • Place Steps: {metadata.get('num_place_steps', 'N/A')}
        • Target Object: {metadata.get('target_object', 'N/A')}
        • Success: {metadata.get('task_success', 'N/A')}
        
        Trajectory Metrics:
        """
        
        if trajectory_data['ee_positions'] is not None:
            ee_pos = trajectory_data['ee_positions']
            path_length = np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1))
            displacement = np.linalg.norm(ee_pos[-1] - ee_pos[0])
            stats_text += f"• Path Length: {path_length:.4f} m\n"
            stats_text += f"• Displacement: {displacement:.4f} m\n"
            stats_text += f"• Path Efficiency: {displacement/path_length:.2%}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 整体标题
        fig.suptitle(f'VLA Trajectory Analysis - Episode {episode_idx}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_multiple_episodes(self, max_episodes: int = 5):
        """
        绘制多个episode的轨迹对比
        
        Args:
            max_episodes: 最多绘制的episode数量
        """
        num_episodes = min(len(self.episodes), max_episodes)
        
        if num_episodes == 0:
            print("No episodes to plot")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
        
        for i in range(num_episodes):
            trajectory_data = self.extract_trajectory_data(self.episodes[i])
            ee_positions = trajectory_data['ee_positions']
            
            if ee_positions is not None:
                x, y, z = ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2]
                ax.plot(x, y, z, label=f'Episode {i}', 
                       color=colors[i], linewidth=2, alpha=0.7)
                ax.scatter(x[0], y[0], z[0], color=colors[i], s=50, marker='o')
                ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=50, marker='s')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Multiple Episodes Trajectory Comparison', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.show()
        return fig
    
    def generate_report(self, output_dir: str = "./trajectory_analysis"):
        """
        生成完整的轨迹分析报告
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating trajectory analysis report in: {output_dir}")
        
        # 为每个episode生成分析图
        for i, episode in enumerate(self.episodes):
            save_path = os.path.join(output_dir, f"episode_{i:03d}_analysis.png")
            self.plot_comprehensive_analysis(episode_idx=i, save_path=save_path)
        
        # 生成多episode对比图
        if len(self.episodes) > 1:
            plt.figure()
            self.plot_multiple_episodes(max_episodes=min(len(self.episodes), 10))
            plt.savefig(os.path.join(output_dir, "multi_episode_comparison.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Report generation complete! Saved {len(self.episodes)} analysis files.")


def main():
    parser = argparse.ArgumentParser(
        description="VLA轨迹可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化整个数据目录
  python visualize_trajectory.py --data-dir ./vla_output
  
  # 可视化单个episode
  python visualize_trajectory.py --episode-file ./vla_output/episode_000.json
  
  # 生成完整报告
  python visualize_trajectory.py --data-dir ./vla_output --generate-report
  
  # 对比多个episodes
  python visualize_trajectory.py --data-dir ./vla_output --compare-episodes 5
        """
    )
    
    parser.add_argument('--data-dir', type=str, 
                       help='数据目录路径')
    parser.add_argument('--episode-file', type=str,
                       help='单个episode JSON文件路径')
    parser.add_argument('--episode-idx', type=int, default=0,
                       help='要可视化的episode索引 (default: 0)')
    parser.add_argument('--generate-report', action='store_true',
                       help='生成完整的分析报告')
    parser.add_argument('--compare-episodes', type=int,
                       help='对比多个episodes（指定数量）')
    parser.add_argument('--output-dir', type=str, default='./trajectory_analysis',
                       help='报告输出目录 (default: ./trajectory_analysis)')
    parser.add_argument('--save', type=str,
                       help='保存图像到指定路径')
    
    args = parser.parse_args()
    
    # 创建可视化器
    try:
        visualizer = TrajectoryVisualizer(
            data_dir=args.data_dir,
            episode_file=args.episode_file
        )
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 执行可视化
    if args.generate_report:
        visualizer.generate_report(output_dir=args.output_dir)
    elif args.compare_episodes:
        visualizer.plot_multiple_episodes(max_episodes=args.compare_episodes)
    else:
        visualizer.plot_comprehensive_analysis(
            episode_idx=args.episode_idx,
            save_path=args.save
        )


if __name__ == "__main__":
    main()

