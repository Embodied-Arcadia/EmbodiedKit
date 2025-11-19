from pxr import Usd, UsdGeom, Gf
from .general import reset_translate_op_safe, get_prims


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

    dx = min(b1.GetMax()[0] - b2.GetMin()[0], b2.GetMax()[0] - b1.GetMin()[0])
    dy = min(b1.GetMax()[1] - b2.GetMin()[1], b2.GetMax()[1] - b1.GetMin()[1])
    dz = min(b1.GetMax()[2] - b2.GetMin()[2], b2.GetMax()[2] - b1.GetMin()[2])

    b1_center = (b1.GetMin() + b1.GetMax()) * 0.5
    b2_center = (b2.GetMin() + b2.GetMax()) * 0.5

    overlap = min(dx, dy)  # min(dx, dy, dz)
    if overlap == dx:
        return Gf.Vec3d(overlap if (b1_center[0] < b2_center[0]) else -overlap, 0, 0)
    elif overlap == dy:
        return Gf.Vec3d(0, overlap if (b1_center[1] < b2_center[1]) else -overlap, 0)
    else:
        return Gf.Vec3d(0, 0, overlap if (b1_center[2] < b2_center[2]) else -overlap)


def is_movable(prim, move_dict):
    """判断物体是否可移动，默认可移动"""
    return move_dict.get(prim, {}).get("movable", True)


def resolve_collisions3(stage, prim_dict, max_iterations=10):
    """
    迭代消除碰撞，保持不可移动物体位置不变。
    prim_dict: {prim: {"movable": True/False}}
    """
    from .general import get_prims_and_move_dict
    prims, move_dict = get_prims_and_move_dict(stage, prim_dict)
    moved = {p: Gf.Vec3d(0, 0, 0) for p in prims}

    for _ in range(max_iterations):
        bboxes = {p: compute_world_bbox(p) for p in prims}
        collision_found = False

        for i, pA in enumerate(prims):
            for pB in prims[i + 1:]:
                mtv = get_overlap_vector(bboxes[pA], bboxes[pB])
                if mtv:
                    collision_found = True
                    movableA = is_movable(pA, move_dict)
                    movableB = is_movable(pB, move_dict)

                    # 根据可移动性分配修正
                    if movableA and movableB:
                        shiftA = -0.5 * mtv
                        shiftB = 0.5 * mtv
                        reset_translate_op_safe(stage, pA, moved[pA] + shiftA)
                        reset_translate_op_safe(stage, pB, moved[pB] + shiftB)
                        moved[pA] += shiftA
                        moved[pB] += shiftB
                    elif movableA:
                        reset_translate_op_safe(stage, pA, moved[pA] - mtv)
                        moved[pA] -= mtv
                    elif movableB:
                        reset_translate_op_safe(stage, pB, moved[pB] + mtv)
                        moved[pB] += mtv
                    # 两个都不可移动，则不动

        if not collision_found:
            break
    else:
        print("达到最大迭代次数，可能仍有少量碰撞")

    return moved
