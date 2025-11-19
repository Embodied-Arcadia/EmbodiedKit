from pxr import Usd, UsdGeom, Gf
import os

from pxr import Usd, UsdGeom, Gf
import numpy as np


def fix_physics_joints_old(stage, offset=Gf.Vec3d(0, 0, 0)):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            for attr_name in ["physics:localPos0", "physics:localPos1"]:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.HasAuthoredValue():
                    pos = attr.Get()
                    new_pos = [pos[i] - offset[i] for i in range(3)]
                    new_pos = Gf.Vec3f(new_pos)
                    attr.Set(new_pos)
                    print(f"Updated {prim.GetName()} {attr_name} to {new_pos}")


def recenter_instance_meshes(stage_path, output_path):
    stage = Usd.Stage.Open(stage_path)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    print(root.GetName())
    for inst in root.GetAllChildren():  # 遍历 Root 下所有 Instance
        if not inst.IsA(UsdGeom.Xform):
            continue
        xform = UsdGeom.Xform(inst)

        # 清理 transform 冲突
        # 获取 transform ops
        ops = xform.GetOrderedXformOps()
        has_scale = any(op.GetOpName() == 'xformOp:scale' for op in ops)
        has_transform = any(op.GetOpName() == 'xformOp:transform' for op in ops)

        if has_scale and has_transform:
            # 保留 scale, 重置 transform 为单位矩阵
            xform_op = xform.GetXformOp('xformOp:transform')
            if xform_op:
                xform_op.Set(Gf.Matrix4d(1.0))

        # 收集所有 Mesh 顶点
        all_points = []
        meshes = []
        parent_mesh_dict = {}
        for mesh in inst.GetAllChildren():
            if mesh.IsA(UsdGeom.Mesh):
                usd_mesh = UsdGeom.Mesh(mesh)
                pts = usd_mesh.GetPointsAttr().Get()
                if pts:
                    points_np = np.array([[p[0], p[1], p[2]] for p in pts])
                    all_points.append(points_np)
                    meshes.append(usd_mesh)
            else:
                # 支持多层 Group
                for submesh in mesh.GetAllChildren():
                    if submesh.IsA(UsdGeom.Mesh):

                        usd_mesh = UsdGeom.Mesh(submesh)
                        pts = usd_mesh.GetPointsAttr().Get()
                        if pts:
                            points_np = np.array([[p[0], p[1], p[2]] for p in pts])
                            all_points.append(points_np)
                            meshes.append(usd_mesh)
                        else:
                            for subsubmesh in submesh.GetAllChildren():
                                if subsubmesh.IsA(UsdGeom.Mesh):
                                    usd_mesh = UsdGeom.Mesh(subsubmesh)
                                    pts = usd_mesh.GetPointsAttr().Get()
                                    if pts:
                                        points_np = np.array([[p[0], p[1], p[2]] for p in pts])
                                        all_points.append(points_np)
                                        meshes.append(usd_mesh)

        if not all_points:
            continue

        # 合并所有顶点，计算中心
        all_points_np = np.vstack(all_points)
        center = all_points_np.mean(axis=0)

        # recover value for physics (minus the center offset)
        #fix_physics_joints(stage, center)

        print(f"Recentering Instance {inst.GetName()}, center={center}")

        # 每个 Mesh 顶点平移
        for mesh in meshes:
            pts = mesh.GetPointsAttr().Get()
            pts_np = np.array([[p[0], p[1], p[2]] for p in pts])
            pts_centered = pts_np - center
            mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pts_centered])
            mesh.GetExtentAttr().Clear()
            pts_centered_min = pts_centered.min(axis=0)
            pts_centered_max = pts_centered.max(axis=0)
            mesh.GetExtentAttr().Set([(pts_centered_min[0], pts_centered_min[1], pts_centered_min[2]),
                                      (pts_centered_max[0], pts_centered_max[1], pts_centered_max[2])])
            parent = mesh.GetPrim().GetParent()
            parent_mesh = UsdGeom.Mesh(parent)
            parent_extent = parent_mesh.GetExtentAttr()
            if not parent_extent.HasAuthoredValue() or parent_extent.Get() is None:
                parent_extent.Set([(pts_centered_min[0], pts_centered_min[1], pts_centered_min[2]),
                                   (pts_centered_max[0], pts_centered_max[1], pts_centered_max[2])])

    # 保存
    stage.GetRootLayer().Export(output_path)
    # 输出 bbox 验证
    # stage = Usd.Stage.Open(output_path)
    # from utils import get_world_bbox, get_bbx_size
    # center, size, minb = get_bbx_size(stage)
    # new_prim = stage.GetDefaultPrim() if stage.HasDefaultPrim() else stage.GetPseudoRoot().GetChildren()[0]
    # center, size, minb = get_world_bbox(stage, new_prim)
    print(f"All Instances recentered and saved to {output_path}")


# 使用
if __name__ == '__main__':
    usd_root = "/home/xianzi/code/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/models/layout/articulated/window/0a7a3099c862986a54c2938fe7c0bdd6/"
    usd_root = "/home/xianzi/code/InternUtopia/internutopia/assets/scenes/GRScenes-100/home_scenes/models/layout/others/ground/0cc228ab61f150b2484bc35a680912cf/"
    recenter_instance_meshes(os.path.join(usd_root, "instance.usd"), os.path.join(usd_root, "instance.usda"))
    # stage = Usd.Stage.Open()
    # prim = stage.GetDefaultPrim()
    # mesh = UsdGeom.Mesh(prim.GetChild("Group_Static").GetChild("MUXAUZAKTKQ5KAABAAAAACI8_primitiveModel_PlankModel_mesh_c9fffd5e__DD__9e60__DD__11ee__DD__b98f__DD__f2d5cc668a6c_roomId_null_0"))
    #
    # points = mesh.GetPointsAttr().Get()
    # import numpy as np
    # points_np = np.array([[p[0], p[1], p[2]] for p in points])
    #
    # center = points_np.mean(axis=0)
    # points_centered = points_np - center
    #
    # mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points_centered])
    # print("Geometry recentered, original center was:", center)
    # stage.GetRootLayer().Export("instance_centered.usd")
