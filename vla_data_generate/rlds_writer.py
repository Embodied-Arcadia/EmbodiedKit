"""
RLDS (Reinforcement Learning Datasets)格式数据写入器
用于将VLA训练数据保存为RLDS格式

RLDS格式参考: https://github.com/google-research/rlds
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import io


class RLDSWriter:
    """RLDS格式数据写入器"""
    
    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        description: str = "VLA robot manipulation dataset",
        version: str = "1.0.0"
    ):
        """
        初始化RLDS写入器
        
        Args:
            dataset_name: 数据集名称
            output_dir: 输出目录
            description: 数据集描述
            version: 数据集版本
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.description = description
        self.version = version
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储episodes
        self.episodes = []
        
        print(f"RLDS Writer initialized: {dataset_name} v{version}")
        print(f"Output directory: {output_dir}")
    
    def add_episode(
        self,
        episode_data: Dict[str, Any],
        episode_id: int
    ):
        """
        添加一个episode到数据集
        
        Args:
            episode_data: episode数据字典
            episode_id: episode ID
        """
        episode_dict = {
            "episode_id": episode_id,
            "steps": episode_data.get("steps", []),
            "metadata": episode_data.get("metadata", {})
        }
        
        self.episodes.append(episode_dict)
        print(f"Added episode {episode_id} with {len(episode_dict['steps'])} steps")
    
    def create_episode_from_trajectory(
        self,
        images: List[np.ndarray],
        actions: List[np.ndarray],
        joint_positions: List[np.ndarray],
        ee_positions: List[np.ndarray],
        ee_orientations: List[np.ndarray],
        instruction: str,
        task_success: bool = True,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        从轨迹数据创建episode
        
        Args:
            images: 图像序列（上帝视角）
            actions: 动作序列
            joint_positions: 关节位置序列
            ee_positions: 末端执行器位置序列
            ee_orientations: 末端执行器姿态序列
            instruction: 任务指令
            task_success: 任务是否成功
            metadata: 额外的元数据
            
        Returns:
            episode数据字典
        """
        steps = []
        
        num_steps = len(images)
        for i in range(num_steps):
            step_data = {
                "observation": {
                    "image": images[i],  # RGB图像 (H, W, 3)
                    "joint_positions": joint_positions[i],  # 关节位置
                    "ee_position": ee_positions[i],  # 末端执行器位置
                    "ee_orientation": ee_orientations[i],  # 末端执行器姿态
                    "instruction": instruction,  # 任务指令
                },
                "action": actions[i] if i < len(actions) else np.zeros_like(actions[0]),
                "reward": 1.0 if (i == num_steps - 1 and task_success) else 0.0,
                "is_first": i == 0,
                "is_last": i == num_steps - 1,
                "is_terminal": i == num_steps - 1 and task_success,
            }
            steps.append(step_data)
        
        episode_metadata = {
            "instruction": instruction,
            "task_success": task_success,
            "num_steps": num_steps,
        }
        if metadata:
            episode_metadata.update(metadata)
        
        return {
            "steps": steps,
            "metadata": episode_metadata
        }
    
    def save_to_tfrecord(self, shard_size: int = 100):
        """
        将episodes保存为TFRecord格式
        
        Args:
            shard_size: 每个shard包含的episode数量
        """
        if not self.episodes:
            print("Warning: No episodes to save!")
            return
        
        # 创建TFRecord目录
        tfrecord_dir = os.path.join(self.output_dir, "tfrecords")
        os.makedirs(tfrecord_dir, exist_ok=True)
        
        # 分shard保存
        num_episodes = len(self.episodes)
        num_shards = (num_episodes + shard_size - 1) // shard_size
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min((shard_id + 1) * shard_size, num_episodes)
            
            shard_filename = os.path.join(
                tfrecord_dir,
                f"{self.dataset_name}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
            )
            
            with tf.io.TFRecordWriter(shard_filename) as writer:
                for episode_idx in range(start_idx, end_idx):
                    episode = self.episodes[episode_idx]
                    
                    # 序列化episode
                    serialized_episode = self._serialize_episode(episode)
                    writer.write(serialized_episode)
            
            print(f"Saved shard {shard_id + 1}/{num_shards}: {shard_filename}")
        
        print(f"Successfully saved {num_episodes} episodes to {tfrecord_dir}")
    
    def save_to_json(self):
        """将episodes保存为JSON格式（用于调试和可视化）"""
        json_path = os.path.join(self.output_dir, f"{self.dataset_name}.json")
        
        # 转换numpy数组为列表
        json_episodes = []
        for episode in self.episodes:
            json_episode = {
                "episode_id": episode["episode_id"],
                "metadata": episode["metadata"],
                "steps": []
            }
            
            for step in episode["steps"]:
                json_step = {
                    "observation": {
                        "joint_positions": step["observation"]["joint_positions"].tolist() 
                            if isinstance(step["observation"]["joint_positions"], np.ndarray) 
                            else step["observation"]["joint_positions"],
                        "ee_position": step["observation"]["ee_position"].tolist()
                            if isinstance(step["observation"]["ee_position"], np.ndarray)
                            else step["observation"]["ee_position"],
                        "ee_orientation": step["observation"]["ee_orientation"].tolist()
                            if isinstance(step["observation"]["ee_orientation"], np.ndarray)
                            else step["observation"]["ee_orientation"],
                        "instruction": step["observation"]["instruction"],
                        "image_shape": step["observation"]["image"].shape if isinstance(step["observation"]["image"], np.ndarray) else None
                    },
                    "action": step["action"].tolist() if isinstance(step["action"], np.ndarray) else step["action"],
                    "reward": float(step["reward"]),
                    "is_first": bool(step["is_first"]),
                    "is_last": bool(step["is_last"]),
                    "is_terminal": bool(step["is_terminal"])
                }
                json_episode["steps"].append(json_step)
            
            json_episodes.append(json_episode)
        
        dataset_info = {
            "name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "num_episodes": len(self.episodes),
            "episodes": json_episodes
        }
        
        with open(json_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Saved dataset metadata to {json_path}")
    
    def _serialize_episode(self, episode: Dict) -> bytes:
        """
        序列化一个episode为TFRecord格式
        
        Args:
            episode: episode数据字典
            
        Returns:
            序列化后的字节串
        """
        # 创建episode的features
        feature_dict = {}
        
        # Episode metadata
        feature_dict["episode_id"] = self._int64_feature(episode["episode_id"])
        feature_dict["num_steps"] = self._int64_feature(len(episode["steps"]))
        
        # Metadata
        for key, value in episode["metadata"].items():
            if isinstance(value, str):
                feature_dict[f"metadata/{key}"] = self._bytes_feature(value.encode('utf-8'))
            elif isinstance(value, (int, bool)):
                feature_dict[f"metadata/{key}"] = self._int64_feature(int(value))
            elif isinstance(value, float):
                feature_dict[f"metadata/{key}"] = self._float_feature(value)
        
        # Steps
        for step_idx, step in enumerate(episode["steps"]):
            prefix = f"steps/{step_idx}"
            
            # Observation
            obs = step["observation"]
            
            # 图像编码为JPEG
            image = obs["image"]
            if isinstance(image, np.ndarray):
                image_encoded = self._encode_image(image)
                feature_dict[f"{prefix}/observation/image"] = self._bytes_feature(image_encoded)
            
            # 关节位置
            feature_dict[f"{prefix}/observation/joint_positions"] = self._float_list_feature(
                obs["joint_positions"]
            )
            
            # 末端执行器位置和姿态
            feature_dict[f"{prefix}/observation/ee_position"] = self._float_list_feature(
                obs["ee_position"]
            )
            feature_dict[f"{prefix}/observation/ee_orientation"] = self._float_list_feature(
                obs["ee_orientation"]
            )
            
            # 指令
            feature_dict[f"{prefix}/observation/instruction"] = self._bytes_feature(
                obs["instruction"].encode('utf-8')
            )
            
            # Action
            feature_dict[f"{prefix}/action"] = self._float_list_feature(step["action"])
            
            # Reward和标志
            feature_dict[f"{prefix}/reward"] = self._float_feature(step["reward"])
            feature_dict[f"{prefix}/is_first"] = self._int64_feature(int(step["is_first"]))
            feature_dict[f"{prefix}/is_last"] = self._int64_feature(int(step["is_last"]))
            feature_dict[f"{prefix}/is_terminal"] = self._int64_feature(int(step["is_terminal"]))
        
        # 创建Example
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()
    
    @staticmethod
    def _encode_image(image: np.ndarray, format: str = 'JPEG', quality: int = 95) -> bytes:
        """将numpy图像编码为JPEG字节串"""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=quality)
        return buffer.getvalue()
    
    @staticmethod
    def _bytes_feature(value):
        """创建bytes feature"""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    @staticmethod
    def _float_feature(value):
        """创建float feature"""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    @staticmethod
    def _int64_feature(value):
        """创建int64 feature"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    @staticmethod
    def _float_list_feature(value):
        """创建float list feature"""
        if isinstance(value, np.ndarray):
            value = value.flatten().tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        total_steps = sum(len(episode["steps"]) for episode in self.episodes)
        
        return {
            "name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "num_episodes": len(self.episodes),
            "total_steps": total_steps,
            "output_dir": self.output_dir
        }


def load_rlds_dataset(tfrecord_pattern: str):
    """
    加载RLDS格式的数据集
    
    Args:
        tfrecord_pattern: TFRecord文件pattern (e.g., "path/to/*.tfrecord")
        
    Returns:
        tf.data.Dataset
    """
    # 定义feature描述
    # 注意：这个需要根据实际保存的数据结构进行调整
    feature_description = {
        'episode_id': tf.io.FixedLenFeature([], tf.int64),
        'num_steps': tf.io.FixedLenFeature([], tf.int64),
    }
    
    def parse_example(serialized_example):
        return tf.io.parse_single_example(serialized_example, feature_description)
    
    # 创建dataset
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_pattern))
    dataset = dataset.map(parse_example)
    
    return dataset

