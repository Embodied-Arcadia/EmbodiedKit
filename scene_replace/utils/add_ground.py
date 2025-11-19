import math

from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf

from .cmp_bbox import get_world_bbox
from .general import reset_translate_op_safe, get_prims


def find_lowest_point(stage, old_prim_path_dict, ignore_ground=True):
    """查找场景中所有物体的最低点(Z坐标最小)"""
    min_z = math.inf
    for key in old_prim_path_dict.keys():
        for prim_path in old_prim_path_dict[key]:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            if ignore_ground and "ground" in prim.GetPath().pathString.lower():
                continue

            if prim.IsA(UsdGeom.Imageable):
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
                bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                bbx_min = bbox.GetMin()
                min_z = min(min_z, bbx_min[2])
    return min_z if min_z != math.inf else 0.0  # 如果没有物体，默认放在Z=0


def create_adaptive_ground(stage, old_prim_path_dict, ground_path="/World/ground", padding=0.0001):
    """
    创建自适应地面，自动位于所有物体下方

    参数:
        stage: USD场景舞台
        ground_path: 地面的Prim路径
        padding: 地面与最低物体的间距
    """
    # 查找最低点
    lowest_z = find_lowest_point(stage, old_prim_path_dict)
    ground_z = lowest_z - padding

    # 计算地面所需大小(基于场景物体范围)
    max_extent = 0.0
    prims = get_prims(stage, old_prim_path_dict)
    for prim in prims:
        if prim.IsA(UsdGeom.Imageable) and "ground" not in prim.GetPath().pathString.lower():
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            max_dim = max(bbox.GetSize())
            max_extent = max(max_extent, max_dim)
    ground_size = max(10.0, max_extent * 2)  # 至少10单位，或物体最大尺寸的2倍

    # 创建XY平面地面
    ground_mesh = UsdGeom.Mesh.Define(stage, f"{ground_path}/mesh")
    half_size = ground_size / 2.0
    ground_mesh.CreatePointsAttr([
        (-half_size, -half_size, ground_z),
        (-half_size, half_size, ground_z),
        (half_size, half_size, ground_z),
        (half_size, -half_size, ground_z)
    ])
    ground_mesh.CreateFaceVertexCountsAttr([4])
    ground_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground_mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    ground_mesh.SetNormalsInterpolation("vertex")

    print(f"创建地面: 位置Z={ground_z}, 大小={ground_size}x{ground_size}")
    return ground_mesh


def setup_physics_for_ground(stage, ground_mesh):
    """为自适应地面设置物理属性"""
    # 物理场景设置(Z轴向下重力)
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # 6. 添加物理API（碰撞 + 刚体 + 网格碰撞）
    UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ground_mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(ground_mesh.GetPrim())

    # 7. 设为静态地板
    ground_mesh.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    ground_mesh.GetPrim().CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    ground_mesh.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")


def align_scene_objects_to_ground(stage, old_prim_path_dict):
    prims = get_prims(stage, old_prim_path_dict)
    for prim in prims:
        # prim_path = str(prim.GetPath())
        # if "window" in prim_path or "curtain" in prim_path:
        #     continue
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
            size, center, bbox_min = get_world_bbox(stage, prim)
            min_z = bbox_min[2]
            lowest_z = min_z
            if lowest_z != 0:
                offset = -lowest_z
                offset_3d = Gf.Vec3d(0, 0, offset)
                reset_translate_op_safe(stage, prim, offset_3d)
