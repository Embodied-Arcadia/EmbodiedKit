from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf


def get_stage_unit_scale(stage):
    """获取 stage 单位（米/单位）"""
    return UsdGeom.GetStageMetersPerUnit(stage)


def get_meters_per_unit(stage):
    """兼容获取场景单位的函数"""
    try:
        # 新版本USD获取方式
        return stage.GetRootLayer().metersPerUnit
    except AttributeError:
        # 旧版本USD回退方案
        return stage.GetMetadata('metersPerUnit') if stage.HasMetadata('metersPerUnit') else 1.0


def get_local_scale_and_matrix(xformable):
    """
    从 UsdGeom.Xformable 提取 local scale 和完整变换矩阵
    """
    local_matrix = xformable.GetLocalTransformation()
    # 分解矩阵为平移、旋转、缩放
    scale = Gf.Vec3d(1.0)
    rotation = Gf.Rotation()
    translation = Gf.Vec3d(0.0)

    # 使用 Gf.Transform 来分解
    transform = Gf.Transform(local_matrix)
    translation = transform.GetTranslation()
    rotation = transform.GetRotation()
    scale = transform.GetScale()

    return scale, local_matrix, translation, rotation


def add_physics_to_prim(prim):
    # 添加刚体（RigidBody）
    rigidAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigidAPI.CreateRigidBodyEnabledAttr(False)  # 静态物体

    # 添加碰撞体（Collision）
    collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)

    # 可选：添加质量属性（默认为自动计算）
    massAPI = UsdPhysics.MassAPI.Apply(prim)
    massAPI.CreateMassAttr(10.0)  # kg

    # 可选：设置物理材质（摩擦、反弹）
    # matPath = prim_path + "_PhysMat"
    # materialPrim = stage.DefinePrim(matPath, "PhysicsMaterial")
    # physxMatAPI = PhysxSchema.PhysxMaterialAPI.Apply(materialPrim)
    # physxMatAPI.CreateStaticFrictionAttr(0.5)
    # physxMatAPI.CreateDynamicFrictionAttr(0.4)
    # physxMatAPI.CreateRestitutionAttr(0.1)

    # 绑定材质到碰撞体
    # collisionAPI.CreateMaterialRel().AddTarget(materialPrim.GetPath())


def reset_translate_op_safe(stage, prim, offset):
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if
                     op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        # 已经有 translate op，直接修改
        current_translate = translate_ops[0].Get()
        new_translate = current_translate + offset  # 只加 Z
        translate_ops[0].Set(new_translate)
    else:
        # 没有 translate op，新增
        xform.AddTranslateOp().Set(offset)


def get_prims(stage, prim_dict):
    prims = []
    for key in prim_dict.keys():
        for prim_path in prim_dict[key]:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            prims.append(prim)
    return prims


def get_prims_and_move_dict(stage, prim_dict):
    prims = []
    move_dict = {}
    for key in prim_dict.keys():
        for prim_path in prim_dict[key]:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            prims.append(prim)
            move_dict[prim] = {"movable": True}
            if "table" in prim_path or "cabinet" in prim_path:
                move_dict[prim] = {"movable": False}
    return prims, move_dict


def reset_rotateZ_safe(xform, angle):
    flag = True
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            flag = False
            attr = op.GetAttr()
            print("当前旋转角度:", attr.Get())
            # 设置新值
            attr.Set(angle)
    if flag:
        xform.AddRotateZOp().Set(angle)
