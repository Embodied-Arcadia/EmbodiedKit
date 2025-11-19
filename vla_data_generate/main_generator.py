import os
import sys
import argparse
import json


def setup_config_files(config_dir: str = "./configs/franka"):
    """
    设置默认配置文件
    
    Args:
        config_dir: 配置文件目录
    """
    print(f"Setting up configuration files in {config_dir}...")
    
    from franka_rrt_controller import create_default_franka_config
    
    robot_desc_path, urdf_path, rrt_config_path = create_default_franka_config(config_dir)
    
    print("Configuration files created:")
    print(f"  - Robot description: {robot_desc_path}")
    print(f"  - URDF: {urdf_path}")
    print(f"  - RRT config: {rrt_config_path}")
    print("\nNote: Please provide a valid Franka URDF file if the default one is not available.")
    
    return robot_desc_path, urdf_path, rrt_config_path


def create_config_template(output_path: str = "./config.json"):
    """
    创建配置文件模板
    
    Args:
        output_path: 输出路径
    """
    config_template = {
        "dataset_name": "vla_franka_manipulation",
        "dataset_description": "VLA robot manipulation dataset with Franka arm using RRT path planning",
        "output_dir": "./vla_output",
        "num_episodes": 10,
        "shard_size": 100,
        
        "robot_description_path": "./configs/franka_description.yaml",
        "urdf_path": "./configs/franka.urdf",
        "rrt_config_path": "./configs/franka_rrt_config.yaml",
        "robot_asset_path": None,
        "robot_start_position": [0.0, 0.0, 0.0],
        
        "scene_usd_path": None,
        
        "objects": [
            {
                "name": "red_cube",
                "type": "cuboid",
                "position": [0.5, 0.0, 0.05],
                "size": [0.05, 0.05, 0.05],
                "color": [1.0, 0.0, 0.0]
            },
            {
                "name": "green_cube",
                "type": "cuboid",
                "position": [0.4, 0.2, 0.05],
                "size": [0.05, 0.05, 0.05],
                "color": [0.0, 1.0, 0.0]
            }
        ],
        
        "tasks": [
            {
                "instruction": "拿起红色方块放到另一处",
                "target_object": "red_cube",
                "target_position": [0.3, 0.3, 0.05]
            },
            {
                "instruction": "Pick up the green cube and place it on the left",
                "target_object": "green_cube",
                "target_position": [-0.2, 0.3, 0.05]
            }
        ],
        
        "camera": {
            "position": [0.0, 0.0, 2.5],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "resolution": [640, 480]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration template created at: {output_path}")


def validate_config(config_path: str) -> bool:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        是否有效
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 检查必需的字段
        required_fields = [
            "robot_description_path",
            "urdf_path",
            "rrt_config_path",
            "output_dir"
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"Error: Missing required fields in config: {missing_fields}")
            return False
        
        # 检查文件是否存在
        file_fields = ["robot_description_path", "urdf_path", "rrt_config_path"]
        for field in file_fields:
            path = config.get(field)
            if path and not os.path.exists(path):
                print(f"Warning: File not found: {path}")
        
        print("Configuration validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return False


def run_generation(config_path: str):
    """
    运行数据生成
    
    Args:
        config_path: 配置文件路径
    """
    print(f"\n{'='*60}")
    print("Starting VLA Data Generation")
    print(f"{'='*60}\n")
    print(f"Using configuration: {config_path}")
    
    # 验证配置
    if not validate_config(config_path):
        print("Configuration validation failed. Aborting.")
        return
    
    # 运行生成器
    from vla_data_generator import run_generation
    
    class Args:
        def __init__(self, config_file):
            self.config_file = config_file
    
    args = Args(config_path)
    run_generation(args)
    
    print(f"\n{'='*60}")
    print("VLA Data Generation Complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="VLA Training Data Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Setup configuration files
        python main_generator.py --setup
        
        # Create configuration template
        python main_generator.py --create-template --output my_config.json
        
        # Validate configuration
        python main_generator.py --validate --config my_config.json
        
        # Generate data
        python main_generator.py --generate --config my_config.json
        
        # All-in-one: setup and generate
        python main_generator.py --setup --generate --config config.json
                """
    )
    
    parser.add_argument("--setup", action="store_true", 
                        help="Setup default configuration files")
    parser.add_argument("--create-template", action="store_true",
                        help="Create configuration template")
    parser.add_argument("--validate", action="store_true",
                        help="Validate configuration file")
    parser.add_argument("--generate", action="store_true",
                        help="Generate VLA training data")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file (default: config.json)")
    parser.add_argument("--output", type=str, default="config.json",
                        help="Output path for template (default: config.json)")
    parser.add_argument("--config-dir", type=str, default="./configs/franka",
                        help="Directory for configuration files (default: ./configs/franka)")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助
    if not any([args.setup, args.create_template, args.validate, args.generate]):
        parser.print_help()
        return
    
    # 执行操作
    if args.setup:
        setup_config_files(args.config_dir)
    
    if args.create_template:
        create_config_template(args.output)
    
    if args.validate:
        validate_config(args.config)
    
    if args.generate:
        run_generation(args.config)


if __name__ == "__main__":
    main()

