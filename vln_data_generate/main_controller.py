# In main_controller.py

import os
import subprocess
import argparse
import json # <-- Import json

"""
Level 1: Main Controller

This script acts as the main entry point for the batch processing workflow.
It performs two main stages:
1.  For each scene, it launches the Level 2 script (path_generator.py) to
    generate and save a collection of navigation paths.
2.  It then reads the generated paths file and launches the Level 3 script
    (robot_navigator.py) for each path to simulate and record data.
"""

def main():
    parser = argparse.ArgumentParser(description="Main controller to process all USD scenes for path generation.")

    parser.add_argument("--isaac_sim_path", type=str, default="../isaac-sim5.0.0",
                        help="Path to the Isaac Sim root directory (e.g., /path/to/isaac-sim).")

    parser.add_argument("--scenes_dir", type=str, default="./assets/scenes", 
                        help="Directory containing the USD scene files.")
                        
    parser.add_argument("--output_dir", type=str, default="./data_output", 
                        help="Directory to save the generated output (trajectories, videos, etc.).")
                        
    parser.add_argument("--num_paths_per_scene", type=int, default=20, 
                        help="Number of random paths to generate for each scene.")
    
    parser.add_argument("--run_mode", type=str, default="run", choices=['test', 'run'],
                        help="Mode to run the script in. 'test' mode only processes the first scene.")

    args = parser.parse_args()

    python_executable = os.path.join(args.isaac_sim_path, "python.sh")

    # --- Input Validation ---
    if not os.path.exists(python_executable):
        print(f"Error: Isaac Sim python executable not found at '{python_executable}'")
        return

    if not os.path.isdir(args.scenes_dir):
        print(f"Error: Scenes directory not found at '{args.scenes_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*50)
    print("Starting Batch Data Generation Process")
    print(f"Scanning for USD files in: {args.scenes_dir}")
    print("="*50)

    # --- Scene Processing Loop ---
    usd_files = sorted(
        f for f in os.listdir(args.scenes_dir)
        if os.path.isfile(os.path.join(args.scenes_dir, f)) and f.lower().endswith(".usd")
    )

    if not usd_files:
        print(f"Error: No USD files found in scenes_dir '{args.scenes_dir}'.")
        return

    for usd_file in usd_files:
        usd_path = os.path.join(args.scenes_dir, usd_file)
        scene_name, _ = os.path.splitext(usd_file)

        print(f"\n--- Processing Scene: {scene_name} ({usd_file}) ---")
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        # === STAGE 1: Generate all paths for the scene ===
        print(f"\n[Stage 1/2] Launching Path Generator for {scene_name}...")
        
        path_gen_command = [
            python_executable,
            os.path.abspath("path_generator.py"),
            "--usd_path", os.path.abspath(usd_path),
            "--output_dir", os.path.abspath(args.output_dir),
            "--num_paths", str(args.num_paths_per_scene),
            "--headless",
        ]

        try:
            subprocess.run(path_gen_command, check=True)
            print(f"Successfully finished path generation for {scene_name}.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error during path generation for {scene_name}. Skipping this scene.")
            print(f"Error details: {e}")
            # continue # Skip to the next scene
            return

        # === STAGE 2: Run navigation simulation for each generated path ===
        print(f"\n[Stage 2/2] Launching Robot Navigator for each path in {scene_name}...")
        
        generated_paths_file = os.path.join(scene_output_dir, "generated_paths.json")
        if not os.path.exists(generated_paths_file):
            print(f"Error: Path generation did not produce the expected file: {generated_paths_file}. Skipping navigation.")
            # continue
            return

        with open(generated_paths_file, 'r') as f:
            all_path_data = json.load(f)

        episodes = all_path_data.get("episodes", [])
        for episode in episodes:
            episode_id = episode["episode_id"]
            
            # Create a dedicated output directory for this specific episode/path
            episode_output_dir = os.path.join(scene_output_dir, f"episode_{episode_id}")
            os.makedirs(episode_output_dir, exist_ok=True)

            # Create a temporary JSON file to pass a single path to the navigator script
            temp_path_file = os.path.join(episode_output_dir, "path_data.json")
            with open(temp_path_file, 'w') as temp_f:
                json.dump({
                    "start_position": episode["start_position"],
                    "goal_position": episode["goals"][0]["position"] if episode.get("goals") else episode.get("goal_position", [0,0,0]),
                    "reference_path": episode["reference_path"],
                    "geodesic_distance": episode.get("info", {}).get("geodesic_distance", 0.0)
                }, temp_f)
            
            print(f"\n--- Running navigation for Episode {episode_id} ---")

            nav_command = [
                python_executable,
                os.path.abspath("robot_navigator.py"),
                "--usd_path", os.path.abspath(usd_path),
                "--path_data_file", os.path.abspath(temp_path_file),
                "--output_dir", os.path.abspath(episode_output_dir),
                "--episode_id", str(episode_id)
            ]
            
            try:
                subprocess.run(nav_command, check=True)
                print(f"Successfully finished navigation for Episode {episode_id}.")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Error during navigation for Episode {episode_id}. Proceeding to next path.")
                print(f"Error details: {e}")

        if args.run_mode == 'test':
            print("\n--- Test mode enabled. Stopping after the first scene. ---")
            break

    print("\n" + "="*50)
    print("Batch processing complete.")
    print("="*50)

if __name__ == "__main__":
    main()