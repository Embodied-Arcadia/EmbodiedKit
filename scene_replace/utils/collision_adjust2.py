from pxr import Usd, UsdGeom, Gf
from .general import reset_translate_op_safe, get_prims

"""
对每一对相交的物体，计算它们的包围盒重叠量。

计算 X/Y/Z 三个方向上的重叠深度。

选择最小的那个作为修正方向（这样微调最小）。

平分这个修正位移，分别应用到两个物体上（保证整体格局更一致）。

迭代直到没有碰撞。
"""


def compute_world_bbox(prim):
    """获取 prim 的世界对齐包围盒 (AABB)"""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return bbox


def get_overlap_vector(b1, b2):
    """计算两个 AABB 的最小修正向量 (MTV)，如果不相交返回 None"""
    if not (b1.GetMax()[0] >= b2.GetMin()[0] and b1.GetMin()[0] <= b2.GetMax()[0] and
            b1.GetMax()[1] >= b2.GetMin()[1] and b1.GetMin()[1] <= b2.GetMax()[1] and
            b1.GetMax()[2] >= b2.GetMin()[2] and b1.GetMin()[2] <= b2.GetMax()[2]):
        return None  # 没有相交

    # X/Y/Z 三个方向上的重叠量
    dx = min(b1.GetMax()[0] - b2.GetMin()[0], b2.GetMax()[0] - b1.GetMin()[0])
    dy = min(b1.GetMax()[1] - b2.GetMin()[1], b2.GetMax()[1] - b1.GetMin()[1])
    dz = min(b1.GetMax()[2] - b2.GetMin()[2], b2.GetMax()[2] - b1.GetMin()[2])

    b1_center = (b1.GetMin() + b1.GetMax()) * 0.5
    b2_center = (b2.GetMin() + b2.GetMax()) * 0.5
    # 找最小修正方向
    overlap = min(dx, dy, dz)
    if overlap == dx:
        return Gf.Vec3d(overlap if (b1_center[0] < b2_center[0]) else -overlap, 0, 0)
    elif overlap == dy:
        return Gf.Vec3d(0, overlap if (b1_center[1] < b2_center[1]) else -overlap, 0)
    else:
        return Gf.Vec3d(0, 0, overlap if (b1_center[2] < b2_center[2]) else -overlap)


def resolve_collisions2(stage, prim_dict, max_iterations=10):
    prims = get_prims(stage, prim_dict)
    moved = {p: Gf.Vec3d(0, 0, 0) for p in prims}

    for _ in range(max_iterations):

        bboxes = {p: compute_world_bbox(p) for p in prims}

        collision_found = False
        for i, pA in enumerate(prims):
            for pB in prims[i + 1:]:
                mtv = get_overlap_vector(bboxes[pA], bboxes[pB])
                if mtv:
                    collision_found = True
                    # 两个物体各移动一半
                    shiftA = -0.5 * mtv
                    shiftB = 0.5 * mtv
                    # xA, xB = UsdGeom.Xformable(pA), UsdGeom.Xformable(pB)
                    # xA.AddTranslateOp().Set(moved[pA] + shiftA)
                    # xB.AddTranslateOp().Set(moved[pB] + shiftB)
                    reset_translate_op_safe(stage, pA, moved[pA] + shiftA)
                    reset_translate_op_safe(stage, pB, moved[pB] + shiftB)
                    moved[pA] += shiftA
                    moved[pB] += shiftB
        if not collision_found:
            break
    else:
        print("达到最大迭代次数，可能仍有少量碰撞")

    return moved
