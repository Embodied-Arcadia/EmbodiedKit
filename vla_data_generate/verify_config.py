#!/usr/bin/env python3
"""
配置文件验证脚本
验证所有必需的配置文件是否存在且格式正确
"""

import os
import sys
import json
import yaml


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}")
        print(f"   路径: {filepath}")
        print(f"   大小: {size} bytes")
        return True
    else:
        print(f"❌ {description}")
        print(f"   路径: {filepath}")
        print(f"   状态: 文件不存在")
        return False


def validate_json(filepath):
    """验证JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return True, data
    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {e}"
    except Exception as e:
        return False, f"读取错误: {e}"


def validate_yaml(filepath):
    """验证YAML文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return True, data
    except yaml.YAMLError as e:
        return False, f"YAML解析错误: {e}"
    except Exception as e:
        return False, f"读取错误: {e}"


def main():
    print("="*70)
    print("VLA数据生成器 - 配置文件验证")
    print("="*70)
    print()
    
    # 定义必需的配置文件
    config_files = {
        "主配置文件": {
            "path": "config_example.json",
            "type": "json",
            "required_keys": ["robot_description_path", "urdf_path", "rrt_config_path"]
        },
        "机器人描述文件": {
            "path": "configs/franka/rmpflow/robot_descriptor.yaml",
            "type": "yaml",
            "required_keys": ["api_version", "cspace", "root_link"]
        },
        "URDF文件": {
            "path": "configs/franka/lula_franka_gen.urdf",
            "type": "urdf",
            "required_keys": None
        },
        "RRT配置文件": {
            "path": "configs/franka/rrt/franka_rrt_config.yaml",
            "type": "yaml",
            "required_keys": ["seed", "step_size", "max_iterations", "task_space_planning_params"]
        }
    }
    
    all_passed = True
    
    # 1. 检查文件存在性
    print("步骤 1: 检查配置文件存在性")
    print("-" * 70)
    
    for name, info in config_files.items():
        filepath = info["path"]
        exists = check_file_exists(filepath, name)
        if not exists:
            all_passed = False
        print()
    
    # 2. 验证文件格式和内容
    print("\n步骤 2: 验证配置文件格式和内容")
    print("-" * 70)
    
    for name, info in config_files.items():
        filepath = info["path"]
        file_type = info["type"]
        required_keys = info["required_keys"]
        
        if not os.path.exists(filepath):
            continue
        
        print(f"\n检查: {name}")
        
        # 验证格式
        if file_type == "json":
            valid, data = validate_json(filepath)
        elif file_type == "yaml":
            valid, data = validate_yaml(filepath)
        elif file_type == "urdf":
            # URDF是XML文件，简单检查是否可读
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                valid = True
                data = None
                if '<robot' in content:
                    print(f"  ✅ URDF格式有效")
                else:
                    print(f"  ⚠️  警告: 可能不是有效的URDF文件")
            except Exception as e:
                valid = False
                data = str(e)
        
        if not valid:
            print(f"  ❌ 格式验证失败: {data}")
            all_passed = False
            continue
        
        if file_type in ["json", "yaml"] and data is not None:
            print(f"  ✅ {file_type.upper()}格式有效")
            
            # 检查必需的键
            if required_keys:
                missing_keys = []
                for key in required_keys:
                    if key not in data:
                        # 对于嵌套键，特殊处理
                        if isinstance(data, dict):
                            missing_keys.append(key)
                
                if missing_keys:
                    print(f"  ⚠️  警告: 缺少以下键: {', '.join(missing_keys)}")
                else:
                    print(f"  ✅ 所有必需键存在")
    
    # 3. 验证主配置中的路径
    print("\n\n步骤 3: 验证主配置文件中的路径")
    print("-" * 70)
    
    main_config_path = "config_example.json"
    if os.path.exists(main_config_path):
        valid, config_data = validate_json(main_config_path)
        if valid:
            paths_to_check = {
                "robot_description_path": "机器人描述文件路径",
                "urdf_path": "URDF文件路径",
                "rrt_config_path": "RRT配置文件路径"
            }
            
            for key, description in paths_to_check.items():
                if key in config_data:
                    path = config_data[key]
                    print(f"\n{description}:")
                    print(f"  配置值: {path}")
                    
                    if path and os.path.exists(path):
                        print(f"  ✅ 文件存在")
                    elif path:
                        print(f"  ❌ 文件不存在")
                        all_passed = False
                    else:
                        print(f"  ⚠️  路径为空或null")
                else:
                    print(f"\n{description}:")
                    print(f"  ❌ 配置中未找到此键")
                    all_passed = False
    
    # 4. 检查RRT配置参数
    print("\n\n步骤 4: 检查RRT配置参数完整性")
    print("-" * 70)
    
    rrt_config_path = "configs/franka/rrt/franka_rrt_config.yaml"
    if os.path.exists(rrt_config_path):
        valid, rrt_data = validate_yaml(rrt_config_path)
        if valid:
            expected_params = {
                "seed": "随机种子",
                "step_size": "步长",
                "max_iterations": "最大迭代次数",
                "max_sampling": "最大采样次数",
                "distance_metric_weights": "距离度量权重",
                "task_space_frame_name": "任务空间框架名称",
                "task_space_limits": "任务空间限制",
                "cuda_tree_params": "CUDA树参数",
                "c_space_planning_params": "C空间规划参数",
                "task_space_planning_params": "任务空间规划参数"
            }
            
            print("\n基础参数:")
            for param, desc in list(expected_params.items())[:7]:
                if param in rrt_data:
                    value = rrt_data[param]
                    if isinstance(value, list):
                        print(f"  ✅ {desc}: {len(value)}个元素")
                    else:
                        print(f"  ✅ {desc}: {value}")
                else:
                    print(f"  ❌ {desc}: 缺失")
                    all_passed = False
            
            print("\n高级参数:")
            for param, desc in list(expected_params.items())[7:]:
                if param in rrt_data:
                    if isinstance(rrt_data[param], dict):
                        num_sub_params = len(rrt_data[param])
                        print(f"  ✅ {desc}: {num_sub_params}个子参数")
                    else:
                        print(f"  ✅ {desc}: 已设置")
                else:
                    print(f"  ❌ {desc}: 缺失")
                    all_passed = False
            
            # 特别检查任务空间规划参数
            if "task_space_planning_params" in rrt_data:
                ts_params = rrt_data["task_space_planning_params"]
                critical_params = [
                    "translation_target_zone_tolerance",
                    "orientation_target_zone_tolerance",
                    "translation_target_final_tolerance",
                    "orientation_target_final_tolerance"
                ]
                
                print("\n  关键任务空间参数:")
                for param in critical_params:
                    if param in ts_params:
                        print(f"    ✅ {param}: {ts_params[param]}")
                    else:
                        print(f"    ⚠️  {param}: 缺失")
    
    # 5. 总结
    print("\n" + "="*70)
    if all_passed:
        print("✅ 配置验证通过！所有必需的配置文件都已正确设置。")
        print("\n下一步:")
        print("  python main_generator.py --generate --config config_example.json")
    else:
        print("❌ 配置验证失败！请检查上述错误并修复。")
        print("\n修复建议:")
        print("  1. 运行: python main_generator.py --setup")
        print("  2. 检查文件路径是否正确")
        print("  3. 确保所有配置文件格式正确")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

