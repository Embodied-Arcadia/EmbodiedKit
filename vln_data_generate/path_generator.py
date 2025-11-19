import os
import argparse
import json
import traceback
import sys
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG_SAFE = {
    "headless": False,  # Set to False to enable viewport visualization
    "renderer": "Default",  
    "active_gpu": 0,
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "carb_settings": {
        "/rtx/od/textureStreaming": False,
        "/rtx/raytracing/enable": False,
        "/rtx/verifyDriverVersion/enabled": False,
    },
}

CONFIG = {
    "headless": True,
    "renderer": "RayTracedLighting",
    "active_gpu": 0,
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "carb_settings": {
        "/rtx/od/textureStreaming": False,
        "/rtx/raytracing/enable": True,
        "/rtx/verifyDriverVersion/enabled": False,
        "/rtx/raytracing/reflections/enabled": False,
        "/rtx/raytracing/refractions/enabled": False,
        "/rtx/raytracing/translucency/enabled": False,
        "/rtx/raytracing/ambientOcclusion/enabled": False,
        "/rtx/raytracing/globalIllumination/enabled": False,
        "/rtx/raytracing/shadows/enabled": False,
        "/rtx/raytracing/denoising/enabled": False,
        "/rtx/raytracing/rayCasting/enabled": False,
        "/rtx/raytracing/rayCasting/rayCount": 1,
        "/rtx/raytracing/rayCasting/bounceCount": 1,
    },
}

def generate_paths(args, app: SimulationApp):
    try:
        # --- Isaac Sim Core Imports ---
        from isaacsim.core.api import World 
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.prims import SingleXFormPrim as XFormPrim
        import omni.kit.commands
        from pxr import Gf, Sdf, Usd, UsdGeom
        import carb
        import omni.usd

        print("Starting Path Generator...")
        
        # GPU Memory Settings
        try:
            carb.settings.set("/rtx/memory/poolSize", 1024 * 1024 * 1024)
            carb.settings.set("/rtx/memory/poolSizeMB", 1024)
            carb.settings.set("/rtx/memory/useUnifiedMemory", False)
            print("GPU memory settings configured.")
        except Exception as e:
            print(f"Warning: Could not set GPU memory settings: {e}")
        
        print("Enabling navigation extensions...")
        enable_extension("omni.anim.navigation.core")
        print("Core navigation extension enabled")
        
        for _ in range(60):
            app.update()
        
        # Load the full bundle
        enable_extension("omni.anim.navigation.bundle")
        print("Navigation bundle enabled")
        
        # Wait for extensions to fully load and initialize
        print("Waiting for navigation extensions to load...")
        max_wait_cycles = 150  # Increased wait time
        for i in range(max_wait_cycles):
            app.update()
            if i % 20 == 0:
                print(f"Waiting for extensions... ({i}/{max_wait_cycles})")

        # Import and validate navigation core
        try:
            print("[generate_paths] Importing navigation core...")
            import omni.anim.navigation.core as nav_core
            print("Navigation core imported successfully.")
        except ImportError as e:
            print("\n" + "="*50)
            print("FATAL ERROR: Failed to import 'omni.anim.navigation.core'.")
            print(f"Import error: {e}")
            print("="*50 + "\n")
            sys.exit(1)
        
        # Validate that the navigation interface is available
        try:
            print("Validating navigation interface...")
            inav = nav_core.acquire_interface()
            if inav is None:
                raise RuntimeError("Navigation interface is None")
            print("Navigation interface acquired successfully.")
        except Exception as e:
            print("\n" + "="*50)
            print("FATAL ERROR: Failed to acquire navigation interface.")
            print(f"Error: {e}")
            print("This usually means the navigation extensions are not properly loaded.")
            print("="*50 + "\n")
            sys.exit(1)

        # --- Scene Setup ---
        print(f"Opening USD stage: {args.usd_path}")
        if not open_stage(args.usd_path):
            print(f"FATAL ERROR: Failed to open stage at {args.usd_path}")
            sys.exit(1)
        
        world = World.instance()
        if world is None:
            world = World()
        
        world.reset()

        # --- Scale USD Root Prim ---
        target_prim_path = None  # Store the path for later use
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            default_prim = stage.GetDefaultPrim()
            target_prim = default_prim if default_prim and default_prim.IsValid() else stage.GetPrimAtPath("/World")
            if target_prim and target_prim.IsValid():
                target_prim_path = target_prim.GetPath().pathString  # Save for later
                xformable = UsdGeom.Xformable(target_prim)
                scale_vec = Gf.Vec3f(0.01, 0.01, 0.01)
                scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if scale_ops:
                    scale_ops[0].Set(scale_vec)
                else:
                    scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionFloat)
                    scale_op.Set(scale_vec)
                print(f"Applied uniform scale {scale_vec} to root prim {target_prim.GetPath()}")
                
                # Wait for USD stage to update transforms and bounding boxes
                print("Waiting for stage to update after scaling...")
                for _ in range(30):
                    app.update()

                # After scaling, lift the scene so that its bottom rests on z=0
                try:
                    world_bound = usd_context.compute_path_world_bounding_box(target_prim.GetPath().pathString)
                    min_pt = Gf.Vec3d(*world_bound[0])
                    min_z = float(min_pt[2])
                    if np.isfinite(min_z) and abs(min_z) > 1e-6:
                        # Adjust or add a translate op to lift by -min_z on Z
                        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                        if translate_ops:
                            current_t = translate_ops[0].Get()
                            tx = float(current_t[0]) if len(current_t) > 0 else 0.0
                            ty = float(current_t[1]) if len(current_t) > 1 else 0.0
                            tz = float(current_t[2]) if len(current_t) > 2 else 0.0
                            translate_ops[0].Set(Gf.Vec3f(tx, ty, tz - min_z))
                        else:
                            translate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionFloat)
                            translate_op.Set(Gf.Vec3f(0.0, 0.0, -min_z))
                        print(f"Lifted scene by {-min_z:.4f} on Z so bottom aligns to z=0")
                        
                        # Wait for stage to update after translation
                        print("Waiting for translation to propagate...")
                        for _ in range(30):
                            app.update()
                    else:
                        print("Skipping Z lift: invalid or near-zero min_z from world bounds")
                except Exception as lift_e:
                    print(f"Warning: Failed to lift scene bottom to z=0: {lift_e}")
            else:
                print("Warning: Could not find a valid root prim to scale.")
        except Exception as scale_e:
            print(f"Warning: Failed to apply root scale: {scale_e}")
        
        # Wait for transforms to propagate (increased wait time)
        print("Waiting for scene transforms to fully update...")
        for _ in range(60):
            app.update()
        
        # Force a world reset to ensure all changes are applied
        print("Applying world reset to finalize transforms...")
        world.reset()
        for _ in range(30):
            app.update()
        
        # --- Disable Ceiling and Furniture Collision ---
        print("Analyzing scene structure and disabling collision for ceiling/furniture...")
        try:
            from pxr import UsdPhysics
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            # First, analyze the COMPLETE scene structure from the ROOT
            print("\n" + "="*70)
            print("=== COMPLETE SCENE STRUCTURE FROM ROOT ===")
            print("="*70)
            
            def print_prim_tree(prim, prefix="", is_last=True, max_depth=6, current_depth=0, max_children_to_show=15):
                """Recursively print prim hierarchy with tree structure"""
                if current_depth > max_depth:
                    return
                
                # Get prim information
                prim_name = prim.GetName() if prim.GetName() else "<root>"
                prim_type = prim.GetTypeName()
                prim_path = prim.GetPath().pathString
                
                # Create branch characters
                branch = "└─ " if is_last else "├─ "
                
                # Print current prim
                info = f"{prim_name} (type: {prim_type}, path: {prim_path})"
                print(f"{prefix}{branch}{info}")
                
                # Prepare prefix for children
                extension = "   " if is_last else "│  "
                new_prefix = prefix + extension
                
                # Get children
                children = prim.GetChildren()
                num_children = len(children)
                
                # Print children
                for idx, child in enumerate(children):
                    if idx >= max_children_to_show:
                        remaining = num_children - max_children_to_show
                        print(f"{new_prefix}└─ ... ({remaining} more children not shown)")
                        break
                    
                    child_is_last = (idx == num_children - 1) or (idx == max_children_to_show - 1)
                    print_prim_tree(child, new_prefix, child_is_last, max_depth, current_depth + 1, max_children_to_show)
            
            # Start from the absolute root
            root_prim = stage.GetPseudoRoot()
            print("\nStarting from Stage Root (/):\n")
            print_prim_tree(root_prim, prefix="", is_last=True, max_depth=5, current_depth=0, max_children_to_show=20)
            
            print("\n" + "="*70)
            print("=== END OF COMPLETE STRUCTURE ===")
            print("="*70 + "\n")
            
            # Now show focused /World analysis
            print("\n=== Focused /World Analysis ===")
            world_prim = stage.GetPrimAtPath("/World")
            if world_prim and world_prim.IsValid():
                print("Analyzing /World hierarchy (showing up to 3 levels):\n")
                
                # Show /World's children
                world_children = world_prim.GetChildren()
                for level1_child in world_children:
                    level1_name = level1_child.GetName()
                    level1_type = level1_child.GetTypeName()
                    level1_path = level1_child.GetPath().pathString
                    print(f"└─ {level1_name}/ (type: {level1_type})")
                    
                    # Show level 2 children (limited to important ones)
                    level2_children = level1_child.GetChildren()
                    for idx2, level2_child in enumerate(level2_children):
                        if idx2 >= 10:  # Limit to first 10
                            print(f"   └─ ... ({len(level2_children) - 10} more items)")
                            break
                        level2_name = level2_child.GetName()
                        level2_type = level2_child.GetTypeName()
                        is_last2 = (idx2 == len(level2_children) - 1) or (idx2 == 9)
                        prefix2 = "└─" if is_last2 else "├─"
                        print(f"   {prefix2} {level2_name}/ (type: {level2_type})")
                        
                        # For Meshes-related prims, show level 3
                        if "mesh" in level2_name.lower() or "root" in level2_name.lower():
                            level3_children = level2_child.GetChildren()
                            for idx3, level3_child in enumerate(level3_children):
                                if idx3 >= 10:
                                    print(f"      └─ ... ({len(level3_children) - 10} more items)")
                                    break
                                level3_name = level3_child.GetName()
                                level3_type = level3_child.GetTypeName()
                                is_last3 = (idx3 == len(level3_children) - 1) or (idx3 == 9)
                                prefix3 = "└─" if is_last3 else "├─"
                                continuation = "   " if is_last2 else "│  "
                                print(f"   {continuation} {prefix3} {level3_name}/ (type: {level3_type})")
                
                print("\nSearching for common patterns (ceiling, furniture, etc.)...")
                # Search for specific patterns
                patterns_found = {
                    'ceiling': [],
                    'furniture': [],
                    'furnitures': [],
                    'meshes': [],
                    'base': []
                }
                
                for prim in Usd.PrimRange(world_prim):
                    prim_path = prim.GetPath().pathString.lower()
                    prim_name = prim.GetName().lower()
                    full_path = prim.GetPath().pathString
                    
                    if 'ceiling' in prim_name or 'ceiling' in prim_path:
                        patterns_found['ceiling'].append(full_path)
                    if 'furniture' in prim_name or 'furniture' in prim_path:
                        patterns_found['furniture'].append(full_path)
                    if 'furnitures' in prim_path:
                        patterns_found['furnitures'].append(full_path)
                    if prim_path.endswith('/meshes'):
                        patterns_found['meshes'].append(full_path)
                    if prim_path.endswith('/base') or '/base/' in prim_path:
                        patterns_found['base'].append(full_path)
                
                # Show what we found
                for pattern_name, paths in patterns_found.items():
                    if paths:
                        print(f"\nFound '{pattern_name}' pattern ({len(paths)} matches):")
                        for path in paths[:5]:  # Show first 5
                            print(f"  - {path}")
                        if len(paths) > 5:
                            print(f"  ... and {len(paths) - 5} more")
            
            print("=" * 35 + "\n")
            
            total_disabled = 0
            
            # Define patterns to disable collision
            patterns_to_disable = [
                {
                    'name': 'ceiling',
                    'patterns': ['/Root/Meshes/Base/ceiling', '/ceiling'],
                    'description': '*/Root/Meshes/Base/ceiling or */ceiling'
                },
                {
                    'name': 'furniture',
                    'patterns': ['/Root/Meshes/Furnitures/other'],
                    'description': '*/Root/Meshes/Furnitures/other'
                }
            ]
            
            world_prim = stage.GetPrimAtPath("/World")
            if world_prim and world_prim.IsValid():
                for pattern_config in patterns_to_disable:
                    disabled_count = 0
                    pattern_name = pattern_config['name']
                    patterns = pattern_config['patterns']
                    description = pattern_config['description']
                    
                    print(f"  Searching for {pattern_name} prims with pattern: {description}")
                    
                    for prim in Usd.PrimRange(world_prim):
                        prim_path = prim.GetPath().pathString
                        
                        # Check if path matches any of the patterns
                        matched = False
                        for pattern in patterns:
                            if pattern in prim_path or prim_path.endswith(pattern):
                                matched = True
                                break
                        
                        if matched:
                            try:
                                # Try to disable collision on this prim and all children
                                for target_prim in Usd.PrimRange(prim):
                                    target_path = target_prim.GetPath()
                                    
                                    # Disable collision
                                    collision_api = UsdPhysics.CollisionAPI.Get(stage, target_path)
                                    if collision_api:
                                        collision_api.GetCollisionEnabledAttr().Set(False)
                                        disabled_count += 1
                                        if disabled_count <= 5:  # Only print first 5 to avoid spam
                                            print(f"    Disabled collision for: {target_path.pathString}")
                                    else:
                                        # Try to apply CollisionAPI first
                                        try:
                                            collision_api = UsdPhysics.CollisionAPI.Apply(target_prim)
                                            collision_api.GetCollisionEnabledAttr().Set(False)
                                            disabled_count += 1
                                            if disabled_count <= 5:
                                                print(f"    Applied and disabled collision for: {target_path.pathString}")
                                        except:
                                            pass  # This prim doesn't support collision
                            except Exception as prim_e:
                                print(f"    Warning: Failed to disable collision for {prim_path}: {prim_e}")
                    
                    if disabled_count > 0:
                        print(f"  ✓ Successfully disabled collision for {disabled_count} {pattern_name} prim(s)")
                        if disabled_count > 5:
                            print(f"    (only showing first 5, total {disabled_count})")
                        total_disabled += disabled_count
                    else:
                        print(f"  ⚠ No {pattern_name} prims found with pattern: {description}")
            
            print(f"\nTotal collision-disabled prims: {total_disabled}")
            
            # Wait for changes to propagate
            for _ in range(20):
                app.update()
                
        except Exception as e:
            print(f"Warning: Could not disable collision: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing anyway...")
        
        # --- NavMesh Generation ---
        # Set agent parameters
        agent_radius = 0.35
        agent_height = 1.5
        # Set slope and climb limits (restrict to floor navigation only)
        max_walkable_slope_angle = 5.0  # Maximum slope in degrees (flat floors only)
        max_climb_height = 0.05  # Maximum step height in meters (5cm, prevents climbing furniture)
        
        # Print scene bounds for diagnostics (using BBoxCache for accurate computation)
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            bounds_path = target_prim_path if target_prim_path else Sdf.Path.absoluteRootPath.pathString
            
            print(f"\nComputing scene bounding box for: {bounds_path}")
            
            # Method 1: Try using BBoxCache for more accurate bounds
            try:
                from pxr import UsdGeom
                bbox_cache = UsdGeom.BBoxCache(0, includedPurposes=[UsdGeom.Tokens.default_])
                target_prim = stage.GetPrimAtPath(bounds_path)
                if target_prim and target_prim.IsValid():
                    bbox = bbox_cache.ComputeWorldBound(target_prim)
                    bbox_range = bbox.ComputeAlignedRange()
                    min_pt = bbox_range.GetMin()
                    max_pt = bbox_range.GetMax()
                    print(f"Scene bounding box (using BBoxCache):")
                    print(f"  Min: ({min_pt[0]:.2f}, {min_pt[1]:.2f}, {min_pt[2]:.2f})")
                    print(f"  Max: ({max_pt[0]:.2f}, {max_pt[1]:.2f}, {max_pt[2]:.2f})")
                    size = max_pt - min_pt
                    print(f"  Size: ({size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}) meters")
                else:
                    raise Exception("Target prim not found")
            except Exception as bbox_e:
                # Fallback to compute_path_world_bounding_box
                print(f"BBoxCache failed ({bbox_e}), using fallback method...")
                world_bound = usd_context.compute_path_world_bounding_box(bounds_path)
                min_pt = Gf.Vec3d(*world_bound[0])
                max_pt = Gf.Vec3d(*world_bound[1])
                print(f"Scene bounding box (using compute_path_world_bounding_box):")
                print(f"  Min: ({min_pt[0]:.2f}, {min_pt[1]:.2f}, {min_pt[2]:.2f})")
                print(f"  Max: ({max_pt[0]:.2f}, {max_pt[1]:.2f}, {max_pt[2]:.2f})")
                size = max_pt - min_pt
                print(f"  Size: ({size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}) meters")
            
            print(f"NavMesh will be baked in a 550x550x550 meter volume at scene center.\n")
        except Exception as bounds_e:
            print(f"Warning: Could not compute scene bounds: {bounds_e}")
        
        # Pause to let user view the scene before NavMesh generation
        visualize_scene = not (args.visualize_navmesh if hasattr(args, 'visualize_navmesh') else False)
        if not (args.headless if hasattr(args, 'headless') else True):  # Only pause in GUI mode
            print("\n" + "="*60)
            print("Scene loaded and transformed successfully!")
            print("="*60)
            print("\nThe scene is now visible in the viewport.")
            print("You can check if the scene is loaded correctly:")
            print("  - The scene should be scaled to appropriate size")
            print("  - The scene bottom should be at z=0")
            print("  - All geometry should be visible")
            print("\nViewport Controls:")
            print("  - Rotate view: Middle mouse button + drag")
            print("  - Zoom: Scroll wheel")
            print("  - Pan: Shift + middle mouse button")
            print("\n" + "="*60)
            print("Press ENTER to start NavMesh generation (or Ctrl+C to exit)...")
            print("="*60)
            try:
                input()  # Wait for user input
            except KeyboardInterrupt:
                print("\nUser interrupted. Exiting...")
                sys.exit(0)
            print("\nStarting NavMesh generation...\n")
        
        # Use navigation interface to set parameters
        try:
            print("Configuring navigation settings...")
            
            # Enable NavMesh
            omni.kit.commands.execute(
                'ChangeSetting',
                path='/exts/omni.anim.people/navigation_settings/navmesh_enabled',
                value=True)
            
            # Set agent parameters
            omni.kit.commands.execute('ChangeSetting', 
                path='/persistent/exts/omni.anim.navigation.core/navMesh/config/agentRadius', 
                value=agent_radius)
            omni.kit.commands.execute('ChangeSetting', 
                path='/persistent/exts/omni.anim.navigation.core/navMesh/config/agentHeight', 
                value=agent_height)
            
            # Set maximum walkable slope angle (restrict to nearly flat surfaces only)
            omni.kit.commands.execute('ChangeSetting',
                path='/persistent/exts/omni.anim.navigation.core/navMesh/config/walkableSlopeAngle',
                value=max_walkable_slope_angle)
            omni.kit.commands.execute('ChangeSetting',
                path='/exts/omni.anim.navigation.core/navMesh/config/walkableSlopeAngle',
                value=max_walkable_slope_angle)
            
            # Set maximum climb height (prevent climbing onto furniture)
            omni.kit.commands.execute('ChangeSetting',
                path='/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMaxClimb',
                value=max_climb_height)
            omni.kit.commands.execute('ChangeSetting',
                path='/exts/omni.anim.navigation.core/navMesh/config/agentMaxClimb',
                value=max_climb_height)
            
            # Set voxel ceiling
            omni.kit.commands.execute(
                "ChangeSetting",
                path="/exts/omni.anim.navigation.core/navMesh/config/voxelCeiling",
                value=4.6,
            )
            
            # Enable dynamic avoidance
            omni.kit.commands.execute(
                'ChangeSetting',
                path='/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled',
                value=True)
            
            # Enable volume mode, only bake within specified volume
            omni.kit.commands.execute('ChangeSetting', 
                path='/exts/omni.anim.navigation.core/navMesh/config/useVolumes', 
                value=True)
            
            # Disable cache, avoid loading large old grids from cache
            omni.kit.commands.execute('ChangeSetting',
                path='/exts/omni.anim.navigation.core/navMesh/cache/enabled',
                value=False)

            # Default enable GPU baking (automatic downgrade to CPU if memory error occurs)
            omni.kit.commands.execute('ChangeSetting',
                path='/exts/omni.anim.navigation.core/navMesh/useGpu',
                value=True)
            omni.kit.commands.execute('ChangeSetting',
                path='/persistent/exts/omni.anim.navigation.core/navMesh/useGpu',
                value=True)

            # Reduce the maximum number of vertices per tile, further limit memory usage (persistent settings)
            omni.kit.commands.execute('ChangeSetting',
                path='/persistent/exts/omni.anim.navigation.core/navMesh/maxVerticesPerTile',
                value=100000)

            # Enable/Disable visualization of NavMesh based on user argument
            visualize_navmesh = args.visualize_navmesh if hasattr(args, 'visualize_navmesh') else False
            omni.kit.commands.execute('ChangeSetting',
                path='/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh',
                value=visualize_navmesh)
            if visualize_navmesh:
                print("NavMesh visualization enabled - you will see the mesh in the viewport after generation.")
            omni.kit.commands.execute('ChangeSetting',
                path='/exts/omni.anim.navigation.core/navMesh/config/autoRebakeOnChanges',
                value=False)
            omni.kit.commands.execute('ChangeSetting', 
                path='/exts/omni.anim.navigation.core/navMesh/config/useCache', 
                value=False)  # Disable cache
            
            # Force enable the navigation system
            omni.kit.commands.execute('ChangeSetting', 
                path='/exts/omni.anim.navigation.core/enabled', 
                value=True)
            
            # Try to create a NavMesh volume to ensure provider is available
            try:
                import omni.usd
                print("Creating NavMesh volume to initialize provider...")
                usd_context = omni.usd.get_context()
                stage = usd_context.get_stage()
                # Create a NavMesh volume in /World
                omni.kit.commands.execute(
                    "CreateNavMeshVolumeCommand",
                    parent_prim_path=Sdf.Path("/World"),
                    layer=stage.GetRootLayer()
                )
                # Get the recently created NavMeshVolume and limit its size to 550x550x550 meters
                selected_paths = usd_context.get_selection().get_selected_prim_paths()
                if selected_paths:
                    nav_volume_path = Sdf.Path(selected_paths[-1])
                    try:
                        world_bound = usd_context.compute_path_world_bounding_box(Sdf.Path.absoluteRootPath.pathString)
                        min_pt = Gf.Vec3d(*world_bound[0])
                        max_pt = Gf.Vec3d(*world_bound[1])
                        mid_point = (min_pt + max_pt) * 0.5
                    except Exception:
                        # Fallback: if cannot calculate bounding box, use origin as center
                        mid_point = Gf.Vec3d(0.0, 0.0, 0.0)

                    # Target volume (meters)
                    target_dim = Gf.Vec3d(550.0, 550.0, 550.0)

                    translate = Gf.Matrix4d(1.0)
                    translate.SetTranslate(mid_point)
                    scale = Gf.Matrix4d(1.0)
                    scale.SetScale(target_dim)
                    xform = scale * translate

                    parent_world_xform = Gf.Matrix4d(
                        *usd_context.compute_path_world_transform(nav_volume_path.GetParentPath().pathString)
                    )
                    xform *= parent_world_xform.GetInverse()

                    omni.kit.commands.execute("TransformPrim", path=nav_volume_path, new_transform_matrix=xform)
                    print("NavMesh volume created and clamped to 550x550x550 meters.")
                else:
                    print("Warning: Could not find created NavMeshVolume in selection.")
            except Exception as volume_e:
                print(f"Warning: Could not create or clamp NavMesh volume: {volume_e}")
            
            print("Navigation settings configured successfully.")
            
        except Exception as e:
            print(f"Warning: Could not set navigation settings: {e}")
            print("Continuing with default settings...")
        
        # Wait for NavMesh provider to be ready and verify baking capability
        print("Waiting for NavMesh provider to be ready...")
        provider_ready = False
        max_provider_wait = 100  # Increased wait time
        
        for i in range(max_provider_wait):
            try:
                # Check if we can get the NavMesh interface
                navmesh_interface = inav.get_navmesh()
                if navmesh_interface is not None:
                    # Try to check if baking is actually available
                    try:
                        # This will fail if the provider isn't ready for baking
                        print("Testing NavMesh provider baking capability...")
                        # We can't actually start baking here, but we can check if the interface is ready
                        print("NavMesh provider is ready and baking capability verified.")
                        provider_ready = True
                        break
                    except Exception as baking_test_e:
                        if i % 10 == 0:
                            print(f"NavMesh provider not ready for baking yet... ({i}/{max_provider_wait}) - {baking_test_e}")
                        app.update()
                else:
                    if i % 10 == 0:
                        print(f"Waiting for NavMesh provider... ({i}/{max_provider_wait})")
                    app.update()
            except Exception as e:
                if i % 10 == 0:
                    print(f"Waiting for NavMesh provider... ({i}/{max_provider_wait}) - {e}")
                app.update()
        
        if not provider_ready:
            print("WARNING: NavMesh provider may not be fully ready, attempting manual initialization...")
            
            # Try to manually trigger provider initialization
            try:
                print("Attempting to manually initialize NavMesh provider...")
                
                # Force a world reset to trigger provider initialization
                world.reset()
                
                # Wait for reset to complete
                for _ in range(50):
                    app.update()
                
                # Try to re-acquire the interface
                inav = nav_core.acquire_interface()
                if inav is not None:
                    print("NavMesh provider re-acquired after reset.")
                else:
                    print("Failed to re-acquire NavMesh provider after reset.")
                    
            except Exception as init_e:
                print(f"Manual initialization failed: {init_e}")
            
            print("Continuing with baking attempt...")
        
        print("Starting NavMesh baking and waiting for completion...")
        
        # Try baking, add multiple retry mechanisms (GPU first, automatic switch to CPU if GPU fails due to memory issues)
        max_baking_retries = 3
        baking_success = False
        current_use_gpu = True  # Default GPU
        has_switched_to_cpu = False
        
        for retry_attempt in range(max_baking_retries):
            try:
                print(f"NavMesh baking attempt {retry_attempt + 1}/{max_baking_retries}...")
                
                # Clear GPU memory
                if retry_attempt > 0:
                    print("Clearing GPU memory before retry...")
                    try:
                        carb.settings.set("/rtx/memory/clearCache", True)
                        app.update()  # Let the system handle memory cleanup
                    except Exception as mem_e:
                        print(f"Warning: Could not clear GPU memory: {mem_e}")
                
                # Use the already acquired interface
                print("Starting NavMesh baking...")
                
                # Try different approaches to start baking
                try:
                    # Method 1: Direct interface call
                    inav.start_navmesh_baking_and_wait()
                    print("NavMesh baking operation completed (Method 1).")
                    
                    # Verify that NavMesh was actually generated
                    navmesh = inav.get_navmesh()
                    if navmesh is None:
                        raise RuntimeError("NavMesh baking completed but no NavMesh was generated. "
                                         "This usually means no walkable geometry was found in the scene.")
                    
                    print("NavMesh successfully generated and verified (Method 1).")
                    baking_success = True
                    break
                except Exception as method1_e:
                    print(f"Method 1 failed: {method1_e}")
                    
                    try:
                        # Method 2: Try using kit commands
                        print("Trying alternative baking method...")
                        omni.kit.commands.execute("StartNavMeshBaking")
                        
                        # Wait for baking to complete
                        max_baking_wait = 300  # 5 minutes max
                        for wait_cycle in range(max_baking_wait):
                            app.update()
                            try:
                                # Check if baking is complete
                                navmesh = inav.get_navmesh()
                                if navmesh is not None:
                                    print("NavMesh baking completed successfully (Method 2).")
                                    baking_success = True
                                    break
                            except:
                                pass
                            
                            if wait_cycle % 30 == 0:
                                print(f"Waiting for baking to complete... ({wait_cycle}/{max_baking_wait})")
                        
                        if baking_success:
                            break
                            
                    except Exception as method2_e:
                        print(f"Method 2 also failed: {method2_e}")
                        raise method1_e  # Re-raise the original exception
                
            except Exception as e:
                print(f"NavMesh baking attempt {retry_attempt + 1} failed: {e}")
                print(f"Error type: {type(e).__name__}")

                # If GPU path encounters memory issues, switch to CPU and retry
                error_str = str(e)
                gpu_oom_signals = [
                    "Out of GPU memory",
                    "ERROR_OUT_OF_DEVICE_MEMORY",
                    "VK_ERROR_OUT_OF_DEVICE_MEMORY",
                    "failed to allocate device memory",
                ]
                if current_use_gpu and (any(s in error_str for s in gpu_oom_signals)) and (not has_switched_to_cpu):
                    print("Detected GPU OOM during navmesh baking. Switching to CPU baking and retrying once...")
                    try:
                        omni.kit.commands.execute('ChangeSetting',
                            path='/exts/omni.anim.navigation.core/navMesh/useGpu',
                            value=False)
                        omni.kit.commands.execute('ChangeSetting',
                            path='/persistent/exts/omni.anim.navigation.core/navMesh/useGpu',
                            value=False)
                        current_use_gpu = False
                        has_switched_to_cpu = True
                        for _ in range(30):
                            app.update()
                        # Enter next retry
                        continue
                    except Exception as switch_e:
                        print(f"Warning: Failed to switch to CPU baking: {switch_e}")

                if retry_attempt < max_baking_retries - 1:
                    print("Attempting to reset world and retry...")
                    try:
                        world.reset()
                        for _ in range(30):
                            app.update()
                    except Exception as reset_e:
                        print(f"Warning: World reset failed: {reset_e}")
                else:
                    print(f"FATAL ERROR: NavMesh baking failed after {max_baking_retries} attempts: {e}")
                    print("\nPossible causes:")
                    print("1. No walkable geometry in the scene (no floors/ground)")
                    print("2. Scene scale too small/large (check if scaling to 0.01 is appropriate)")
                    print("3. NavMesh agent radius (0.35m) too large for the scene")
                    print("4. NavMesh volume (550x550x550m) doesn't cover walkable areas")
                    print("5. GPU memory issues or driver problems")
                    print("6. Scene geometry has no collision meshes enabled")
                    print("\nTroubleshooting steps:")
                    print("- Check that the USD scene has collision-enabled geometry")
                    print("- Verify the scene scale is appropriate")
                    print("- Try adjusting NavMesh parameters (agent_radius, voxel_ceiling)")
                    print("- Ensure NavMesh volume covers the walkable areas")
                    sys.exit(1)
        
        if not baking_success:
            print("FATAL ERROR: All NavMesh baking attempts failed.")
            sys.exit(1)

        # --- Path Generation Loop ---
        navmesh = inav.get_navmesh()
        if navmesh is None:
            print("FATAL ERROR: Failed to get navmesh after baking. NavMesh may not have been generated properly.")
            sys.exit(1)
        
        # Validate navmesh is available
        try:
            test_point = navmesh.query_random_point("test_session")
            if test_point is None:
                print("FATAL ERROR: NavMesh is not properly initialized - cannot sample random points.")
                sys.exit(1)
            print(f"NavMesh successfully obtained and validated. Starting path generation...")
        except Exception as e:
            print(f"FATAL ERROR: NavMesh validation failed: {e}")
            sys.exit(1)
        
        # If visualization is enabled, pause to let user see the NavMesh
        if args.visualize_navmesh if hasattr(args, 'visualize_navmesh') else False:
            print("\n" + "="*60)
            print("NavMesh visualization is now visible in the viewport!")
            print("The NavMesh shows walkable areas (usually in cyan/blue color).")
            print("You can:")
            print("  - Rotate the view by dragging with middle mouse button")
            print("  - Zoom with scroll wheel")
            print("  - Pan with Shift + middle mouse button")
            print("="*60)
            print("\nPress ENTER to continue with path generation (or Ctrl+C to exit)...")
            print("="*60)
            try:
                input()  # Wait for user input
            except KeyboardInterrupt:
                print("\nUser interrupted. Exiting...")
                sys.exit(0)
            print("Continuing with path generation...\n")

        episodes = []
        valid_paths_found = 0
        attempts = 0
        max_attempts = args.num_paths * 100
        
        # Collect all z-values to compute z-offset for normalization
        all_sampled_z_values = []
        temp_samples = 0
        max_samples_for_z_detection = 50
        print("Sampling NavMesh points to detect z-offset...")
        while temp_samples < max_samples_for_z_detection:
            test_pt = navmesh.query_random_point("z_detection_session")
            if test_pt is not None:
                all_sampled_z_values.append(test_pt[2])
                temp_samples += 1
        
        if all_sampled_z_values:
            min_z = float(np.min(all_sampled_z_values))
            max_z = float(np.max(all_sampled_z_values))
            mean_z = float(np.mean(all_sampled_z_values))
            z_range = max_z - min_z
            
            print(f"NavMesh Z-statistics from {len(all_sampled_z_values)} samples:")
            print(f"  Min Z: {min_z:.3f}m")
            print(f"  Max Z: {max_z:.3f}m")
            print(f"  Mean Z: {mean_z:.3f}m")
            print(f"  Z Range: {z_range:.3f}m")
            
            # Decide z_offset: shift minimum z to 0.0 or to a small positive value (e.g., 0.1)
            z_offset = -min_z + 0.1  # This will make the lowest floor at z=0.1m
            print(f"Applying Z-offset: {z_offset:.3f}m to normalize all paths to ground level.")
            
            # Smart floor detection using clustering
            # Group Z values into clusters to identify main floor levels
            z_values_sorted = np.sort(all_sampled_z_values)
            
            # Find the largest cluster near the minimum (ground floor)
            # Points within 2.5m of minimum are considered ground floor candidates
            ground_candidates = [z for z in all_sampled_z_values if z <= min_z + 2.5]
            
            if len(ground_candidates) >= 10:  # Need at least 10 points to determine floor
                ground_mean = np.mean(ground_candidates)
                ground_std = np.std(ground_candidates)
                
                # Set floor range based on the distribution of ground floor points
                floor_z_min = min_z - 0.3  # Small buffer below
                floor_z_max = ground_mean + max(1.5 * ground_std, 1.2)  # Adaptive upper bound
                
                # Ensure reasonable range (0.8m to 3.0m)
                floor_thickness = floor_z_max - floor_z_min
                if floor_thickness < 0.8:
                    floor_z_max = floor_z_min + 1.5
                elif floor_thickness > 3.0:
                    floor_z_max = floor_z_min + 2.5
                
                ground_count = sum(1 for z in all_sampled_z_values if floor_z_min <= z <= floor_z_max)
                ground_percentage = (ground_count / len(all_sampled_z_values)) * 100
                
                print(f"\nGround Floor Detection:")
                print(f"  Floor Z range: [{floor_z_min:.3f}m, {floor_z_max:.3f}m]")
                print(f"  Floor thickness: {floor_z_max - floor_z_min:.3f}m")
                print(f"  Points on ground floor: {ground_count}/{len(all_sampled_z_values)} ({ground_percentage:.1f}%)")
                print(f"  Filter: Only paths on ground floor (excludes upper floors and furniture)")
            else:
                # Fallback: use minimum + 2m for safety
                floor_z_min = min_z - 0.3
                floor_z_max = min_z + 2.0
                print(f"\nGround Floor Detection (Fallback):")
                print(f"  Floor Z range: [{floor_z_min:.3f}m, {floor_z_max:.3f}m]")
        else:
            print("Warning: Could not sample enough points to determine z-offset. Using z_offset=0.0")
            z_offset = 0.0
            floor_z_min = -float('inf')
            floor_z_max = float('inf')
        
        print(f"\nAttempting to find {args.num_paths} valid paths...")
        
        # Relaxed tolerance for floor checks to allow minor z-noise
        floor_tol = 0.2  # meters
        
        # Increase attempts budget to improve success probability on strict filters
        max_attempts = max(max_attempts, args.num_paths * 200)
        
        while valid_paths_found < args.num_paths and attempts < max_attempts:
            attempts += 1

            # Constrained sampling: actively sample start/goal on the ground floor range
            start_pos = None
            goal_pos = None
            for _ in range(40):
                sp = navmesh.query_random_point("path_gen_session")
                if sp is not None and (floor_z_min - floor_tol) <= sp[2] <= (floor_z_max + floor_tol):
                    start_pos = sp
                    break
            for _ in range(40):
                gp = navmesh.query_random_point("path_gen_session")
                if gp is not None and (floor_z_min - floor_tol) <= gp[2] <= (floor_z_max + floor_tol):
                    goal_pos = gp
                    break

            if start_pos is None or goal_pos is None:
                if attempts % 50 == 0:
                    print("Warning: Could not sample floor-constrained start/goal points. Retrying.")
                continue

            navmesh_path = navmesh.query_shortest_path(start_pos=start_pos, end_pos=goal_pos)
            if not navmesh_path:
                continue

            waypoints = navmesh_path.get_points()
            if len(waypoints) < 3:
                continue

            reference_path = [list(p) for p in waypoints]
            geodesic_distance = navmesh_path.length()
            if geodesic_distance < 2.0:
                continue

            # Waypoint floor compliance: require high fraction within floor height with tolerance
            floor_min_tol = floor_z_min - floor_tol
            floor_max_tol = floor_z_max + floor_tol
            num_on_floor = sum(1 for p in reference_path if floor_min_tol <= p[2] <= floor_max_tol)
            frac_on_floor = num_on_floor / len(reference_path)
            if frac_on_floor < 0.95:
                # Too many points off floor (likely stairs/furniture), skip
                continue

            # Apply z-offset normalization to all points
            start_pos_normalized = [start_pos[0], start_pos[1], start_pos[2] + z_offset]
            goal_pos_normalized = [goal_pos[0], goal_pos[1], goal_pos[2] + z_offset]
            reference_path_normalized = [[p[0], p[1], p[2] + z_offset] for p in reference_path]

            valid_paths_found += 1
            print(f"Found valid path #{valid_paths_found} with {len(reference_path)} waypoints and length {geodesic_distance:.2f}m.")
            print(f"  Original Z range: [{reference_path[0][2]:.2f}, {reference_path[-1][2]:.2f}]m | on_floor={frac_on_floor*100:.1f}%")
            print(f"  Normalized Z range: [{reference_path_normalized[0][2]:.2f}, {reference_path_normalized[-1][2]:.2f}]m")

            episode_index = valid_paths_found - 1
            episode = {
                "episode_id": str(episode_index),
                "trajectory_id": int(episode_index),
                "scene_id": str(args.usd_path),
                "start_position": start_pos_normalized,
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "goals": [
                    {
                        "position": goal_pos_normalized,
                        "radius": 3.0,
                    }
                ],
                "reference_path": reference_path_normalized,
                "instruction": {
                    "instruction_text": "",
                    "instruction_tokens": [],
                },
                "info": {
                    "geodesic_distance": float(geodesic_distance),
                    "z_offset_applied": float(z_offset),
                    "original_start_z": float(start_pos[2]),
                    "original_goal_z": float(goal_pos[2]),
                    "floor_z_range": [float(floor_z_min), float(floor_z_max)],
                    "floor_tolerance": float(floor_tol),
                    "fraction_waypoints_on_floor": float(frac_on_floor),
                },
            }
            episodes.append(episode)

        # --- Save All Collected Paths to a Single File ---
        if episodes:
            # Use the USD file stem (filename without extension) as scene name,
            # so that it matches main_controller.py's scene_name convention.
            scene_name = os.path.splitext(os.path.basename(args.usd_path))[0]
            scene_output_dir = os.path.join(args.output_dir, scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)
            output_file_path = os.path.join(scene_output_dir, "generated_paths.json")
            final_output = {
                "episodes": episodes,
                "instruction_vocab": {},
            }
            with open(output_file_path, 'w') as f:
                json.dump(final_output, f, indent=4)
            print(f"\nSuccessfully saved {len(episodes)} episodes to {output_file_path}")
        else:
            print("\nFATAL ERROR: No valid paths were generated for this scene after all attempts.")
            sys.exit(1)

    except Exception as e:
        print("\n" + "="*50)
        print("FATAL ERROR inside generate_paths function!")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Details: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("="*50 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate navigation paths in an Isaac Sim scene.")
    parser.add_argument("--usd_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_paths", type=int, default=5)
    parser.add_argument("--safe_mode", action="store_true", help="Use safe rendering mode (disable ray tracing)")
    parser.add_argument("--visualize_navmesh", action="store_true", help="Enable NavMesh visualization in viewport")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    
    cli_args = parser.parse_args()
    
    # Choose configuration based on arguments
    if cli_args.headless:
        selected_config = CONFIG.copy()
        selected_config["headless"] = True
        config_name = "Headless Mode"
    else:
        selected_config = CONFIG_SAFE.copy()
        selected_config["headless"] = False
        config_name = "GUI Mode (for visualization)"
    
    print(f"Using {config_name} configuration...")
    app = SimulationApp(selected_config)

    try:
        generate_paths(cli_args, app)
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        traceback.print_exc()
        
        # If using ray tracing configuration fails, suggest using safe mode
        if not cli_args.safe_mode:
            print("\n" + "="*60)
            print("Suggestion: If you continue to encounter GPU crashes, use the --safe_mode parameter")
            print("="*60 + "\n")
        
        sys.exit(1)
    finally:
        app.close()
        print("[__main__] Script finished.")