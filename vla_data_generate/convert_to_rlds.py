"""
将原始数据转换为RLDS格式的独立脚本
此脚本应在安装了TensorFlow的环境中运行，与Isaac Sim环境分离

使用方法:
    python convert_to_rlds.py --input_dir ./vla_output --output_dir ./vla_rlds
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List
from pathlib import Path


def load_episode_data(episode_dir: str) -> Dict:
    """
    从目录加载单个episode的原始数据
    
    Args:
        episode_dir: episode目录路径
        
    Returns:
        episode数据字典
    """
    print(f"Loading episode from {episode_dir}...")
    
    # 加载元数据
    with open(os.path.join(episode_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 加载图像
    images_dir = os.path.join(episode_dir, "images")
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    images = [np.load(os.path.join(images_dir, f)) for f in image_files]
    
    # 加载动作和状态
    actions = np.load(os.path.join(episode_dir, "actions.npy"))
    joint_positions = np.load(os.path.join(episode_dir, "joint_positions.npy"))
    ee_positions = np.load(os.path.join(episode_dir, "ee_positions.npy"))
    ee_orientations = np.load(os.path.join(episode_dir, "ee_orientations.npy"))
    
    episode_data = {
        "episode_id": metadata["episode_id"],
        "instruction": metadata["instruction"],
        "images": images,
        "actions": actions,
        "joint_positions": joint_positions,
        "ee_positions": ee_positions,
        "ee_orientations": ee_orientations,
        "task_success": metadata["task_success"],
        "metadata": metadata["metadata"]
    }
    
    print(f"  Loaded {len(images)} steps")
    return episode_data


def convert_to_rlds(input_dir: str, output_dir: str):
    """
    将原始数据转换为RLDS格式
    
    Args:
        input_dir: 原始数据目录
        output_dir: RLDS输出目录
    """
    print("="*60)
    print("Converting Raw Data to RLDS Format")
    print("="*60)
    
    # 导入rlds_writer（需要TensorFlow）
    try:
        from rlds_writer import RLDSWriter
    except ImportError as e:
        print("Error: Cannot import rlds_writer. Make sure TensorFlow is installed.")
        print(f"  {e}")
        return
    
    # 加载数据集信息
    dataset_info_path = os.path.join(input_dir, "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        print(f"Error: Dataset info not found at {dataset_info_path}")
        return
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    print(f"\nDataset: {dataset_info['dataset_name']}")
    print(f"Episodes: {dataset_info['num_episodes']}")
    
    # 创建RLDS写入器
    writer = RLDSWriter(
        dataset_name=dataset_info.get("dataset_name", "vla_franka_manipulation"),
        output_dir=output_dir,
        description=dataset_info.get("description", "VLA robot manipulation dataset")
    )
    
    # 查找所有episode目录
    episode_dirs = sorted([
        d for d in os.listdir(input_dir) 
        if d.startswith("episode_") and os.path.isdir(os.path.join(input_dir, d))
    ])
    
    print(f"\nFound {len(episode_dirs)} episodes to convert")
    
    # 转换每个episode
    for episode_dirname in episode_dirs:
        episode_dir = os.path.join(input_dir, episode_dirname)
        
        try:
            # 加载原始数据
            episode_data = load_episode_data(episode_dir)
            
            # 转换为RLDS格式
            rlds_episode = writer.create_episode_from_trajectory(
                images=episode_data["images"],
                actions=list(episode_data["actions"]),
                joint_positions=list(episode_data["joint_positions"]),
                ee_positions=list(episode_data["ee_positions"]),
                ee_orientations=list(episode_data["ee_orientations"]),
                instruction=episode_data["instruction"],
                task_success=episode_data["task_success"],
                metadata=episode_data["metadata"]
            )
            
            # 添加到writer
            writer.add_episode(rlds_episode, episode_data["episode_id"])
            print(f"  Converted episode {episode_data['episode_id']}")
            
        except Exception as e:
            print(f"  Error converting {episode_dirname}: {e}")
            continue
    
    # 保存RLDS数据集
    print("\nSaving RLDS dataset...")
    writer.save_to_json()
    
    # 如果需要TFRecord格式
    try:
        shard_size = dataset_info.get("config", {}).get("shard_size", 100)
        writer.save_to_tfrecord(shard_size=shard_size)
        print("  TFRecord format saved")
    except Exception as e:
        print(f"  Warning: Could not save TFRecord format: {e}")
    
    # 打印信息
    rlds_info = writer.get_dataset_info()
    print("\n" + "="*60)
    print("RLDS Conversion Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    for key, value in rlds_info.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw VLA data to RLDS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python convert_to_rlds.py --input_dir ./vla_output --output_dir ./vla_rlds
    
Note: This script requires TensorFlow. Run in a separate environment from Isaac Sim.
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw episode data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for RLDS format data"
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行转换
    convert_to_rlds(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

