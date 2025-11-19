# robot_navigator.py (同步版本)

import os
import argparse
import json
import numpy as np
import carb
from omni.isaac.kit import SimulationApp
import imageio
import traceback

# ... (CONFIG and SimplePathFollower class remain unchanged) ...
CONFIG = {
    "headless": True,
    "renderer": "RayTracedLighting",
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
}

class SimplePathFollower:
    # ... (No changes needed in this class) ...
    def __init__(self, robot_articulation, path, linear_speed=0.5, angular_speed=1.0):
        self.robot = robot_articulation
        self.path = [np.array(p) for p in path]
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.target_waypoint_idx = 1
        self.path_completed = False
    
    def get_command(self):
        if self.path_completed:
            return None, None, "IDLE"

        current_pos, current_rot_quat = self.robot.get_world_pose()
        target_pos = self.path[self.target_waypoint_idx]
        
        direction_vector = target_pos - current_pos
        distance_to_target = np.linalg.norm(direction_vector)
        
        if distance_to_target < 0.2:
            self.target_waypoint_idx += 1
            if self.target_waypoint_idx >= len(self.path):
                self.path_completed = True
                print("Path completed!")
                return 0.0, 0.0, "STOP"
            target_pos = self.path[self.target_waypoint_idx]
            direction_vector = target_pos - current_pos

        forward_vec = carb.math.quat_to_rot_matrix(current_rot_quat.tolist())[:, 0]
        
        target_dir_normalized = direction_vector / np.linalg.norm(direction_vector)
        dot_product = np.dot(forward_vec[:2], target_dir_normalized[:2])
        angle_to_target = np.arccos(np.clip(dot_product, -1.0, 1.0))

        cross_product = np.cross(forward_vec, target_dir_normalized)
        turn_direction = -1.0 if cross_product[2] < 0 else 1.0
        
        angular_vel = 0.0
        action = "MOVE_FORWARD"
        if angle_to_target > 0.1:
            angular_vel = self.angular_speed * turn_direction
            action = "TURN_LEFT" if turn_direction > 0 else "TURN_RIGHT"
        
        linear_vel = self.linear_speed if angle_to_target < np.pi / 4 else 0.0
        
        return linear_vel, angular_vel, action


def run_navigation(args):
    # --- Isaac Sim Core and Asset Imports ---
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.sensors.camera import Camera
    # Prefer new Articulation API per api.md; fallback to legacy if unavailable
    IS_NEW_ARTICULATION_API = True
    try:
        from isaacsim.core.prims import Articulation as ArticulationView
    except ImportError:
        from omni.isaac.core.articulations import Articulation as ArticulationView
        IS_NEW_ARTICULATION_API = False

    # --- Load Path Data ---
    with open(args.path_data_file, 'r') as f:
        path_data = json.load(f)

    # --- Debug: Validate loaded path data ---
    print(f"[DEBUG] Loaded path data file: {os.path.abspath(args.path_data_file)}")
    print(f"[DEBUG] Keys present: {list(path_data.keys())}")
    start_position = np.array(path_data.get("start_position", [0, 0, 0]))
    goal_position = np.array(path_data.get("goal_position", [0, 0, 0]))
    reference_path = path_data.get("reference_path", [])
    geodesic_distance = path_data.get("geodesic_distance", 0.0)
    print(f"[DEBUG] start_position: {start_position.tolist()}")
    print(f"[DEBUG] goal_position: {goal_position.tolist()}")
    print(f"[DEBUG] reference_path points: {len(reference_path)} | geodesic_distance: {geodesic_distance}")
    
    # --- World and Scene Setup ---
    world = World.instance()
    if world is None:
        try:
            world = World(stage_units_in_meters=1.0)
        except Exception:
            world = World()
    # Open stage with robust fallback
    try:
        if hasattr(world, "scene") and hasattr(world.scene, "open"):
            world.scene.open(args.usd_path)
        else:
            raise AttributeError("World.scene.open not available")
    except Exception as e:
        print(f"[WARN] world.scene.open failed: {e}. Trying fallback open_stage...")
        try:
            import isaacsim.core.utils.stage as stage_utils
            if not stage_utils.open_stage(args.usd_path):
                raise RuntimeError(f"open_stage failed for {args.usd_path}")
        except Exception as e2:
            raise RuntimeError(f"Failed to open USD stage: {e2}")
    world.reset()
    print(f"[DEBUG] USD opened: {os.path.abspath(args.usd_path)}")
    
    # --- Add G1 Robot to the Scene ---
    robot_asset_path = "./assets/g1_29dof_color_camera.usd"
    robot_prim_path = "/World/g1"
    create_prim(prim_path=robot_prim_path, prim_type="Xform", position=start_position, usd_path=robot_asset_path)
    # Create Articulation per documented API
    if IS_NEW_ARTICULATION_API:
        robot = ArticulationView(prim_paths_expr=robot_prim_path, name="g1_view")
    else:
        robot = ArticulationView(robot_prim_path)
    world.scene.add(robot)
    # Initialize and validate articulation per api.md
    try:
        robot.initialize()
    except Exception as e:
        print(f"Warning: Articulation.initialize() failed: {e}")
    try:
        if hasattr(robot, "is_valid") and not robot.is_valid():
            print("Warning: Articulation is not valid for prim path:", robot_prim_path)
    except Exception as e:
        print(f"Warning: Articulation validity check failed: {e}")
    
    # --- Add Camera to Robot Torso ---
    camera = None
    try:
        camera_prim_path = f"{robot_prim_path}/base_link/torso_camera"
        camera = Camera(prim_path=camera_prim_path, name="torso_camera", position=np.array([0.2, 0.0, 0.5]), resolution=(640, 480))
        camera.initialize()
        print(f"[DEBUG] Camera initialized at: {camera_prim_path} | resolution: (640, 480)")
    except Exception as e:
        print(f"Warning: Failed to initialize camera: {e}")
        camera = None
    
    world.reset() # 再次reset以确保所有对象都已初始化
    print("Robot added to the scene" + (" with camera." if camera else " (no camera)."))
    
    # --- Initialize Controller and Data Recorders ---
    controller = SimplePathFollower(robot, reference_path)
    agent_path_recorder = []
    video_frames = []
    frames_captured = 0
    first_frame_logged = False
    steps_executed = 0
    
    # --- Simulation Loop ---
    world.play()
    for i in range(2000): # Max simulation steps
        world.step(render=True) # 同步执行一步，必须render才能捕获视频
        steps_executed += 1
        
        linear_vel, angular_vel, action = controller.get_command()
        
        if controller.path_completed:
            print("Simulation finished.")
            break
            
        # Motion control per API: prefer Articulation.set_velocities (new API)
        try:
            if IS_NEW_ARTICULATION_API and hasattr(robot, "set_velocities"):
                # Map forward linear velocity to x, yaw rate to angular z for the first articulation
                velocities = np.zeros((getattr(robot, "count", 1), 6), dtype=float)
                velocities[0, 0] = float(linear_vel or 0.0)  # vx
                velocities[0, 5] = float(angular_vel or 0.0)  # wz
                robot.set_velocities(velocities)
            else:
                # Legacy fallback: preserve previous behavior
                robot.apply_action(np.array([linear_vel, angular_vel]))
        except Exception as ctrl_e:
            if steps_executed % 50 == 0:
                print(f"[WARN] Failed to apply motion control at step {steps_executed}: {ctrl_e}")
        
        current_pos, current_rot = robot.get_world_pose()
        agent_path_recorder.append({
            "position": current_pos.tolist(),
            "rotation": current_rot.tolist(),
            "action": action
        })
        if camera is not None:
            try:
                frame = camera.get_rgba()[:, :, :3]
                video_frames.append(frame)
                frames_captured += 1
                if not first_frame_logged:
                    print(f"[DEBUG] First frame captured with shape: {frame.shape}")
                    first_frame_logged = True
            except Exception as cam_e:
                if i % 50 == 0:
                    print(f"[WARN] Failed to capture camera frame at step {i}: {cam_e}")

        if i % 50 == 0:
            print(f"[DEBUG] step={i} | waypoint_idx={controller.target_waypoint_idx} | action={action} | frames={frames_captured}")
    else:
        print("Warning: Simulation ended due to reaching max steps (2000).")

    world.stop()

    # --- Save All Recorded Data ---
    # ... (This entire section remains unchanged) ...
    print("Saving recorded data...")
    os.makedirs(args.output_dir, exist_ok=True)
    nav_task_info = {
        "episode_id": args.episode_id,
        "scene_id": os.path.basename(os.path.dirname(args.usd_path)),
        "start_position": start_position.tolist(),
        "start_rotation": [0,0,0,1],
        "goals": [{"position": goal_position.tolist(), "radius": 0.5}],
        "reference_path": reference_path,
        "info": {"geodesic_distance": geodesic_distance}
    }
    nav_task_file = os.path.join(args.output_dir, "navigation_task.json")
    with open(nav_task_file, 'w') as f:
        json.dump(nav_task_info, f, indent=4)
    nav_task_file_abs = os.path.abspath(nav_task_file)
    nav_task_ok = os.path.exists(nav_task_file_abs) and os.path.getsize(nav_task_file_abs) > 0
    print(f"Saved navigation task info to: {nav_task_file_abs} | exists={nav_task_ok} | size={os.path.getsize(nav_task_file_abs) if os.path.exists(nav_task_file_abs) else 0}")

    agent_trajectory_file = os.path.join(args.output_dir, "agent_trajectory.json")
    with open(agent_trajectory_file, 'w') as f:
        json.dump({"trajectory": agent_path_recorder}, f, indent=4)
    agent_trajectory_file_abs = os.path.abspath(agent_trajectory_file)
    traj_ok = os.path.exists(agent_trajectory_file_abs) and os.path.getsize(agent_trajectory_file_abs) > 0
    print(f"Saved agent trajectory to: {agent_trajectory_file_abs} | len={len(agent_path_recorder)} | exists={traj_ok} | size={os.path.getsize(agent_trajectory_file_abs) if os.path.exists(agent_trajectory_file_abs) else 0}")

    if video_frames:
        video_file = os.path.join(args.output_dir, "trajectory_video.mp4")
        with imageio.get_writer(video_file, fps=30) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        video_file_abs = os.path.abspath(video_file)
        video_ok = os.path.exists(video_file_abs) and os.path.getsize(video_file_abs) > 0
        print(f"Saved video to: {video_file_abs} | frames={len(video_frames)} | exists={video_ok} | size={os.path.getsize(video_file_abs) if os.path.exists(video_file_abs) else 0}")
    else:
        print("[WARN] No video frames captured. Video was not saved.")

    # --- Debug: Final summary ---
    print("[DEBUG] Summary:")
    print(f"  steps_executed={steps_executed}")
    print(f"  waypoints_in_path={len(reference_path)} | path_completed={controller.path_completed}")
    print(f"  frames_captured={frames_captured}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Navigate a robot along a path and record data.")
    parser.add_argument("--usd_path", type=str, required=True)
    parser.add_argument("--path_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    
    cli_args = parser.parse_args()

    app = SimulationApp(CONFIG)

    try:
        run_navigation(cli_args)
    except Exception as e:
        print(f"An error occurred during navigation: {e}")
        traceback.print_exc()
    finally:
        app.close()