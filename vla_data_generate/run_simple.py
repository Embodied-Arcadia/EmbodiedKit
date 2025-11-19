#!/usr/bin/env python3
"""
使用 Isaac Sim 运行 VLA 数据生成器
"""

import os
import sys
import subprocess
import argparse

def find_isaac_sim():
    possible_paths = [
        "../isaac-sim4.5.0",
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.5.0"),
    ]
    
    for path in possible_paths:
        python_sh = os.path.join(path, "python.sh")
        if os.path.exists(python_sh):
            return path
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="VLA 数据生成器启动器 - 使用 Isaac Sim Python",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--isaac-sim",
        type=str,
        default=None,
        help="Isaac Sim 安装路径（自动检测如果未指定）"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config_example.json",
        help="配置文件路径（默认: config_example.json）"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="验证配置文件"
    )
    
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="创建配置模板"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="设置配置文件"
    )
    
    args = parser.parse_args()
    
    # 查找 Isaac Sim
    isaac_sim_path = args.isaac_sim
    if isaac_sim_path is None:
        print("🔍 自动查找 Isaac Sim 安装...")
        isaac_sim_path = find_isaac_sim()
        
        if isaac_sim_path is None:
            print("\n❌ 错误: 未找到 Isaac Sim 安装")
            print("\n请手动指定路径:")
            print("  python run_simple.py --isaac-sim /path/to/isaac-sim")
            print("\n或者检查以下路径是否存在:")
            print("  - ../isaac-sim4.5.0")
            print("  - ~/.local/share/ov/pkg/isaac-sim-*")
            sys.exit(1)
    
    # 验证 Python 可执行文件
    python_executable = os.path.join(isaac_sim_path, "python.sh")
    
    if not os.path.exists(python_executable):
        print(f"\n❌ 错误: 找不到 Isaac Sim python.sh")
        print(f"路径: {python_executable}")
        sys.exit(1)
    
    print(f"✅ 使用 Isaac Sim: {isaac_sim_path}")
    print(f"✅ Python: {python_executable}")
    print()
    
    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), "main_generator.py")
    
    cmd = [python_executable, script_path]
    
    # 添加参数
    if args.validate:
        cmd.extend(["--validate", "--config", args.config])
    elif args.create_template:
        cmd.extend(["--create-template", "--output", args.config])
    elif args.setup:
        cmd.append("--setup")
    else:
        # 默认: 生成数据
        cmd.extend(["--generate", "--config", args.config])
    
    # 显示命令
    print("=" * 60)
    print("运行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    print()
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=False)
        
        print()
        print("=" * 60)
        if result.returncode == 0:
            print("✅ 执行成功")
            
            # 显示输出
            output_dir = "vla_output"
            if os.path.isdir(output_dir):
                print(f"\n📂 输出目录: {output_dir}")
                print("\n文件列表:")
                for root, dirs, files in os.walk(output_dir):
                    level = root.replace(output_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f'{indent}{os.path.basename(root)}/')
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
                        print(f'{subindent}{file} ({size_str})')
        else:
            print(f"❌ 执行失败 (退出码: {result.returncode})")
        print("=" * 60)
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

