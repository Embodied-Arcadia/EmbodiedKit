import json
import random
import os
import argparse

from pxr import Usd, UsdGeom, Gf

from utils import reset_rotateZ_safe
from utils import get_world_bbox
from utils import get_stage_unit_scale
from utils import get_local_scale_and_matrix
from utils import add_physics_to_prim
from utils import create_adaptive_ground, setup_physics_for_ground, align_scene_objects_to_ground
from utils import resolve_collisions3
from utils import recenter_instance_meshes


def search_random_new_usd_path(old_key, new_usd_json_dic):
    for key1 in new_usd_json_dic.keys():
        for key2 in new_usd_json_dic[key1].keys():
            if old_key in new_usd_json_dic[key1][key2].keys():
                target_key1 = None
                for key3 in new_usd_json_dic[key1][key2].keys():
                    if old_key in key3:
                        target_key1 = key3
                        break
                target_root = new_usd_json_dic[key1][key2][target_key1]
                target_key2 = random.choice(list(os.listdir(target_root)))
                return os.path.join(target_root, target_key2, "instance.usd")
    return False


def replace_prim_with_model(stage, old_prim_path_dict, new_usd_json_dic, new_prim_subpath, usd_root, save_usd_path):
    to_delete_prim = []

    for old_key, old_prim_path_to_target in old_prim_path_dict.items():
        for old_prim_path, new_usd_path in old_prim_path_to_target.items():
            # get old object prim
            old_prim = stage.GetPrimAtPath(old_prim_path)

            if not old_prim:
                raise ValueError(f"old prim path {old_prim_path} not exists!")
            old_xformable = UsdGeom.Xformable(old_prim)
            scale, local_matrix, translation, rotation = get_local_scale_and_matrix(old_xformable)
            l_size, l_center, _ = get_world_bbox(stage, old_prim)
            reset_rotateZ_safe(old_xformable, 0)
            # original scene unit
            old_unit = get_stage_unit_scale(stage)

            # get bbox size for original object
            old_size, old_center, _ = get_world_bbox(stage, old_prim)

            if old_size[0] < 1e-3 or old_size[1] < 1e-3 or old_size[2] < 1e-3:
                to_delete_prim.append(old_prim_path)
            if new_usd_path == "":
                new_usd_path = search_random_new_usd_path(old_key, new_usd_json_dic)
            else:
                new_usd_path = os.path.join(usd_root, new_usd_path)
            if not new_usd_path or not os.path.exists(new_usd_path):
                raise ValueError(f"Can not find a usd instance for replacement, please check the category '{old_key}'!")
            new_stage = Usd.Stage.Open(new_usd_path)

            new_prim = new_stage.GetDefaultPrim() if new_stage.HasDefaultPrim() else \
                new_stage.GetPseudoRoot().GetChildren()[0]
            new_size, new_center, new_min_box = get_world_bbox(new_stage, new_prim)
            if abs(new_size[0]) > 1e6:
                new_base_dir = os.path.dirname(new_usd_path)
                name = os.path.basename(new_usd_path)
                if ".usda" in name:
                    name = name[:-1]
                new_usda_path = f"{new_base_dir}/{name}a"
                recenter_instance_meshes(new_usd_path, new_usda_path)
                new_stage = Usd.Stage.Open(new_usda_path)
                new_usd_path = new_usda_path
                new_prim = new_stage.GetDefaultPrim() if new_stage.HasDefaultPrim() else \
                    new_stage.GetPseudoRoot().GetChildren()[0]
                new_size, new_center, new_min_box = get_world_bbox(new_stage, new_prim)

            if not new_prim:
                raise ValueError(f"new prim path {new_prim_subpath} which is in {new_usd_path} does not exist!")

            # new unit for new scene
            new_unit = get_stage_unit_scale(new_stage)

            # unit conversion
            unit_scale = old_unit / new_unit

            # compute scale factor（Scale proportionally based on the smallest edge, or change to max/avg）
            scale_factor = min(
                old_size[0] / (new_size[0]),
                old_size[1] / (new_size[1]),
                old_size[2] / (new_size[2]),
            )
            swap_new_size = new_size / unit_scale
            xy_swapped = (abs(old_size[0] - swap_new_size[1]) + abs(old_size[1] - swap_new_size[0])) < abs(
                old_size[0] - swap_new_size[0]) + abs(old_size[1] - swap_new_size[1])

            if xy_swapped:
                print("swap prim name: ", old_prim_path)
                print("scale factor before swapping:", scale_factor)
                # new_size, new_center, new_min_box = get_world_bbox(new_stage, new_prim)
                scale_factor = min(
                    old_size[0] / (new_size[1]),
                    old_size[1] / (new_size[0]),
                    old_size[2] / (new_size[2]),
                )
                print("scale factor before swapping:", scale_factor)

            print(f"[INFO] old unit: {old_unit} m/unit")
            print(f"[INFO] new unit: {new_unit} m/unit")
            print(f"[INFO] unit scale: {unit_scale}")
            print(f"[INFO] scale factor: {scale_factor}")
            # print(f"[INFO] 中心偏移: {center_offset}")

            # delete old prim
            stage.RemovePrim(old_prim_path)

            # create new Xform container
            new_xform = UsdGeom.Xform.Define(stage, old_prim_path)
            new_xform.GetPrim().GetReferences().AddReference(new_usd_path)
            add_physics_to_prim(new_xform.GetPrim())

            # isaacsim need to obey this order (translate, orientation, scale)
            new_xform.AddTranslateOp().Set(translation)
            reset_angle = 0
            if "chair" in old_prim_path:
                reset_angle = random.choice([-180])
                reset_rotateZ_safe(new_xform, reset_angle)

            if xy_swapped:
                # recover the direction
                reset_angle += 90
                reset_rotateZ_safe(new_xform, reset_angle)
                # new_xform.AddRotateZOp().Set(90.0)
            new_xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
            new_xform.AddScaleOp().Set(Gf.Vec3f(scale_factor))

    # delete prim
    for prim_path in to_delete_prim:
        stage.RemovePrim(prim_path)
    # create adaptive ground
    align_scene_objects_to_ground(stage, old_prim_path_dict)
    ground_mesh = create_adaptive_ground(stage, old_prim_path_dict)
    setup_physics_for_ground(stage, ground_mesh)

    # collision detection
    moved = resolve_collisions3(stage, old_prim_path_dict)
    print(moved)

    # stage.GetRootLayer().Save()
    stage.GetRootLayer().Export(save_usd_path)
    print("[INFO] replace complete！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert USD materials with assets")
    parser.add_argument("--usd_path", type=str, required=True,
                        help="usd file needed to be replaced.")
    parser.add_argument("--scene_root", type=str, required=True,
                        help="scene dir of InternUtopia you have used, e.g, "
                             "./internutopia/assets/scenes/GRScenes-100/home_scenes")

    parser.add_argument("--save_usd_path", type=str, required=True,
                        help="usd file path for saving replaced scene")

    args = parser.parse_args()

    usd_path = args.usd_path
    scene_root = args.scene_root
    stage = Usd.Stage.Open(usd_path)
    if "bounding_box" in usd_path:
        stage.SetMetadata("metersPerUnit", 1.0)

    old_prim_path_dict = json.load(open("./prim_match_dict.json", "r"))
    new_usd_json_dic = json.load(open("./home_scenes.json"))

    replace_prim_with_model(
        stage=stage,
        old_prim_path_dict=old_prim_path_dict,
        new_usd_json_dic=new_usd_json_dic,
        new_prim_subpath="",
        usd_root=scene_root,
        save_usd_path=args.save_usd_path
    )
