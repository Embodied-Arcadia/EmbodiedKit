import os
import json
import re
import argparse
from pxr import Usd
from tqdm import tqdm

from utils import get_world_bbox, recenter_instance_meshes, get_stage_unit_scale


def set_symlink(target_dir):
    for root, dirs, files in tqdm(os.walk(os.path.join(target_dir, 'models'))):
        for _ in files:
            link_name = os.path.join(root, 'Materials')
            if os.path.exists(link_name):
                os.remove(link_name)
            target_material_dir = os.path.join(target_dir, 'Materials')
            target_material_dir_relpath = os.path.relpath(target_material_dir, root)
            os.symlink(target_material_dir_relpath, link_name)

    # create "Materials" and "models" symlink with stage start_result_interaction.usd
    for root, dirs, files in tqdm(os.walk(os.path.join(target_dir, 'scenes'))):
        for _ in files:
            materials_link_name = os.path.join(root, 'Materials')
            models_link_name = os.path.join(root, 'models')
            if os.path.exists(materials_link_name):
                os.remove(materials_link_name)
            if os.path.exists(models_link_name):
                os.remove(models_link_name)
            target_material_dir = os.path.join(target_dir, 'Materials')
            target_material_dir_relpath = os.path.relpath(target_material_dir, root)
            target_model_dir = os.path.join(target_dir, 'models')
            target_model_dir_relpath = os.path.relpath(target_model_dir, root)
            if not os.path.islink(materials_link_name):
                os.symlink(target_material_dir_relpath, materials_link_name)
            if not os.path.islink(models_link_name):
                os.symlink(target_model_dir_relpath, models_link_name)


def find_texture_path_in_mdl(mdl_file_path):
    with open(mdl_file_path, 'r') as file:
        lines = file.readlines()
    pattern = re.compile(r'texture_2d\("([^"]+)",')
    result = set()
    for line in lines:
        match = re.findall(pattern, line)
        if match:
            result.update(match)

    return result


def get_obj_size(usd_path):
    stage = Usd.Stage.Open(usd_path)
    unit = get_stage_unit_scale(stage)
    prim = stage.GetDefaultPrim() if stage.HasDefaultPrim() else \
        stage.GetPseudoRoot().GetChildren()[0]
    size, center, min_box = get_world_bbox(stage, prim)
    if abs(size[0]) > 100000:
        base_dir = os.path.dirname(usd_path)
        name = os.path.basename(usd_path)
        new_usda_path = f"{base_dir}/{name}a"
        recenter_instance_meshes(usd_path, new_usda_path)
        new_stage = Usd.Stage.Open(new_usda_path)
        new_prim = new_stage.GetDefaultPrim() if new_stage.HasDefaultPrim() else \
            new_stage.GetPseudoRoot().GetChildren()[0]
        size, center, min_box = get_world_bbox(new_stage, new_prim)
        usd_path = new_usda_path
    size *= unit
    return size, usd_path


def get_material_reference(stage, prim):
    material_refs = {}
    material_bindings = prim.GetRelationship('material:binding').GetTargets()
    for material_path in material_bindings:
        material_prim = stage.GetPrimAtPath(material_path)
        if material_prim:
            for shader in material_prim.GetChildren():
                for shader_property in shader.GetProperties():
                    if shader_property.GetTypeName() == 'asset':
                        mtl_name = shader_property.GetName()
                        if mtl_name not in material_refs:
                            material_refs[mtl_name] = str(shader_property.Get().path)
                shader_mdl_absolute_path = shader.GetProperty('info:mdl:sourceAsset').Get().resolvedPath
                if shader_mdl_absolute_path:
                    texture_path_set = find_texture_path_in_mdl(shader_mdl_absolute_path)
                    material_refs["mdl"] = [os.path.join('./Materials', texture_path) for texture_path in
                                            texture_path_set]
    return material_refs


def get_detail_content(usd_path):
    size, usd_path = get_obj_size(usd_path)
    list_size = [round(float(x), 5) for x in size]
    stage = Usd.Stage.Open(usd_path)
    relative_path = usd_path.split('/')[-5:]
    instance = {"usd_path": os.path.join("./models", "/".join(relative_path)), "size": list_size}

    material = {}
    component_num = 0
    texture_num = 0
    for prim in stage.Traverse():
        if prim.HasRelationship('material:binding'):
            material_ref = get_material_reference(stage, prim)

            material[f"component_{component_num}"] = material_ref
            component_num += 1
        if prim.GetName() == 'textures':
            for attr in prim.GetAttributes():
                if attr.GetTypeName() == 'asset':
                    material[f"texture_{texture_num}"] = str(attr.Get().path)

    instance["material"] = material
    return instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert USD materials with assets")
    parser.add_argument("--scene_root", type=str, required=True,
                        help="scene root dir of InternUtopia. e.g, "
                             "./internutopia/assets/scenes/GRScenes-100/home_scenes/")

    args = parser.parse_args()
    root = args.scene_root
    model_root = os.path.join(root, "models")
    set_symlink(root)
    json_dict = {}
    for dir in os.listdir(model_root):
        json_dict[dir] = {}
        for type in os.listdir(os.path.join(model_root, dir)):
            json_dict[dir][type] = {}
            for cate in tqdm(os.listdir(os.path.join(model_root, dir, type))):
                if "layout" in dir and cate not in ["cabinet", "window"]:
                    continue
                json_dict[dir][type][cate] = os.path.join(model_root, dir, type, cate)
                for instance in tqdm(os.listdir(os.path.join(model_root, dir, type, cate))):
                    usd_path = os.path.join(model_root, dir, type, cate, instance, "instance.usd")
                    save_path = os.path.join(model_root, dir, type, cate, instance, "base_info.json")
                    if os.path.exists(save_path):
                        continue
                    base_info = get_detail_content(usd_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    json.dump(base_info, open(save_path, "w", encoding='utf-8'), indent=4)
    print(json_dict)
    json.dump(json_dict, open('./home_scenes.json', 'w', encoding='utf-8'), indent=4)
