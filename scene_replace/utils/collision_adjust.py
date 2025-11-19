from pxr import Usd, UsdGeom, Gf
from .general import reset_translate_op_safe, get_prims


def compute_world_bbox(prim):
    """获取 prim 的世界对齐包围盒 (AABB)"""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return bbox


def bbox_intersects(b1, b2):
    """判断两个 AABB 是否相交"""
    return (b1.GetMax()[0] >= b2.GetMin()[0] and b1.GetMin()[0] <= b2.GetMax()[0] and
            b1.GetMax()[1] >= b2.GetMin()[1] and b1.GetMin()[1] <= b2.GetMax()[1] and
            b1.GetMax()[2] >= b2.GetMin()[2] and b1.GetMin()[2] <= b2.GetMax()[2])


def separate_objects(stage, prim_dict, max_iterations=10, separation_axis=Gf.Vec3d(1, 0, 0)):
    """
    消解碰撞: 将相交的物体沿 separation_axis 平移开
    - stage: Usd.Stage
    - prims: list of Usd.Prim
    - max_iterations: 最大迭代次数，避免死循环
    - separation_axis: 平移方向（默认 X 轴）
    """
    prims = get_prims(stage, prim_dict)
    moved = {p: Gf.Vec3d(0, 0, 0) for p in prims}  # 记录平移累计值

    for _ in range(max_iterations):
        bboxes = {p: compute_world_bbox(p) for p in prims}

        collision_found = False
        for i, pA in enumerate(prims):
            for pB in prims[i + 1:]:
                if bbox_intersects(bboxes[pA], bboxes[pB]):
                    collision_found = True
                    # 计算推开距离（在分离轴上）
                    overlap = bboxes[pA].GetMax()[0] - bboxes[pB].GetMin()[0]
                    if overlap <= 0:  # 已分离
                        continue

                    shift = separation_axis.GetNormalized() * (overlap + 0.01)  # 加一点 margin
                    # 移动 pB
                    # xform = UsdGeom.Xformable(pB)
                    # xform.AddTranslateOp().Set(shift + moved[pB])
                    reset_translate_op_safe(stage, pB, shift + moved[pB])
                    moved[pB] += shift

        if not collision_found:
            break  # 没有碰撞了，结束
    else:
        print("达到最大迭代次数，可能仍有碰撞")

    return moved
