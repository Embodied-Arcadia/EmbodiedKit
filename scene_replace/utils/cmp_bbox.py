import numpy as np

from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf

def compute_bbox_safe(stage, prim):
    """安全计算 prim 的 bbox，如果 Xform 出现 inf 就递归找 Mesh"""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()

    if np.isinf(bbox_range.GetMin()[0]):  # inf 的情况
        all_points = []

        def collect_points(p):
            for child in p.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(child)
                    pts = mesh.GetPointsAttr().Get()
                    if pts:
                        all_points.extend(pts)
                collect_points(child)

        collect_points(prim)
        if not all_points:
            return None
        arr = np.array(all_points)
        min_pt, max_pt = arr.min(axis=0), arr.max(axis=0)
        return min_pt, max_pt
    else:
        return bbox_range.GetMin(), bbox_range.GetMax()


def get_world_bbox(stage, prim):
    """获取 prim 在世界空间下的包围盒和中心点"""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    size = bbox.GetSize()
    # new_size = bbox.GetMax() - bbox.GetMin()
    # new_size = Gf.Vec3f(abs(size[0]), abs(size[1]), abs(size[2]))
    center = (bbox.GetMin() + bbox.GetMax()) * 0.5
    return size, center, bbox.GetMin()


from pxr import UsdGeom, Gf


def get_local_bbox(stage, prim):
    """获取 prim 在局部空间下的包围盒和中心点"""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
    size = bbox.GetSize()
    center = (bbox.GetMin() + bbox.GetMax()) * 0.5
    return size, center, bbox.GetMin()


def get_bbx_size(stage):
    all_points = []
    scale = np.array([1.0, 1.0, 1.0])
    flag = True
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform) and flag:
            xform = UsdGeom.Xform(prim)
            ops = xform.GetOrderedXformOps()
            for op in ops:
                if op.GetOpName() == 'xformOp:scale':
                    scale_val = op.Get()  # 返回 GfVec3d
                    scale = np.array([scale_val[0], scale_val[1], scale_val[2]])
                    flag = False
        if prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            pts = usd_mesh.GetPointsAttr().Get()
            if pts:
                points_np = np.array([[p[0], p[1], p[2]] for p in pts])
                all_points.append(points_np)

    all_points_np = np.vstack(all_points)
    min_pt = all_points_np.min(axis=0)
    max_pt = all_points_np.max(axis=0)
    center = all_points_np.mean(axis=0)
    center = (min_pt + max_pt) / 2
    size = max_pt - min_pt
    size *= scale
    # 不知道为啥，要换一下顺序
    new_size = [size[1], size[0], size[2]]
    print("BBox center:", center)
    print("BBox size:", new_size)
    return new_size, center, min_pt * scale


def get_bbx_size_from_prim(prim):
    all_points = []
    if prim.IsA(UsdGeom.Mesh):
        usd_mesh = UsdGeom.Mesh(prim)
        pts = usd_mesh.GetPointsAttr().Get()
        if pts:
            points_np = np.array([[p[0], p[1], p[2]] for p in pts])
            all_points.append(points_np)
    scale = np.array([1.0, 1.0, 1.0])
    flag = True
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Xform) and flag:
            xform = UsdGeom.Xform(child)
            ops = xform.GetOrderedXformOps()
            for op in ops:
                if op.GetOpName() == 'xformOp:scale':
                    scale_val = op.Get()  # 返回 GfVec3d
                    scale = np.array([scale_val[0], scale_val[1], scale_val[2]])
                    flag = False

    def traverse_prim(prim, all_points):
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Mesh):
                usd_mesh = UsdGeom.Mesh(child)
                pts = usd_mesh.GetPointsAttr().Get()
                if pts:
                    points_np = np.array([[p[0], p[1], p[2]] for p in pts])
                    all_points.append(points_np)
            all_points = traverse_prim(child, all_points)
        return all_points

    all_points = traverse_prim(prim, all_points)
    all_points_np = np.vstack(all_points)
    min_pt = all_points_np.min(axis=0)
    max_pt = all_points_np.max(axis=0)
    center = (min_pt + max_pt) / 2
    size = max_pt - min_pt
    size *= scale
    print("BBox center:", center)
    print("BBox size:", size)
    return size, center
