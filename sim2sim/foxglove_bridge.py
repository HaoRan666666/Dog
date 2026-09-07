"""Foxglove 实时数据发布桥：把 MuJoCo 状态发布成 Foxglove topic。

独立于 sim2sim 部署脚本，供 lab2mujoco2.py 等入口按需调用：
    bridge = FoxgloveBridge(m, d, host=..., port=..., enabled=...)
    bridge.publish()
"""

import time

import numpy as np

import mujoco

# foxglove-sdk 是可选依赖：未安装时不应阻塞 sim2sim 启动。
# 用 try/except 判定，未安装时 FChan/FMsg 置 None；启用 FoxgloveBridge 时在 __init__ 里自动降级为禁用。
try:
    import foxglove
    from foxglove import channels as FChan
    from foxglove import messages as FMsg

    _HAS_FOXGLOVE = True
except ImportError:
    foxglove = None
    FChan = None
    FMsg = None
    _HAS_FOXGLOVE = False

# ── MuJoCo 关节顺序 (XML body 遍历顺序) ─────────────────────────────
# LF: ABAD, HIP, KENN, FOOT  →  mj_idx 0,1,2,3
# RF: ABAD, HIP, KENN, FOOT  →  mj_idx 4,5,6,7
# LB: ABAD, HIP, KENN, FOOT  →  mj_idx 8,9,10,11
# RB: ABAD, HIP, KENN, FOOT  →  mj_idx 12,13,14,15
MJ_JOINT_NAMES = [
    "LF_ABAD_JOINT", "LF_HIP_JOINT", "LF_KENN_JOINT", "LF_FOOT_JOINT",
    "RF_ABAD_JOINT", "RF_HIP_JOINT", "RF_KENN_JOINT", "RF_FOOT_JOINT",
    "LB_ABAD_JOINT", "LB_HIP_JOINT", "LB_KENN_JOINT", "LB_FOOT_JOINT",
    "RB_ABAD_JOINT", "RB_HIP_JOINT", "RB_KENN_JOINT", "RB_FOOT_JOINT",
]


class FoxgloveBridge:
    """把 MuJoCo 状态发布成 Foxglove topic。

    topic:
      /tf           foxglove.FrameTransform  机身 + 4 足端 (3D 面板)
      /joint_states foxglove.JointStates     16 关节数组 (URDF/3D 用)
      /joints/<NAME> foxglove.JointState    每关节一个 topic, Plot 面板按名选择
      /imu/orientation foxglove.Quaternion  姿态四元数 (xyzw)
      /imu/angular_velocity foxglove.Vector3 角速度
      /imu/linear_acceleration foxglove.Vector3 线加速度
      /imu/projected_gravity foxglove.Vector3 投影重力 (机体系单位向量)
      /scene        foxglove.SceneUpdate     简易骨架 (base 立方体 + 4 条腿折线)
      /contacts     foxglove.SceneUpdate     接触点球体 (颜色/大小=接触力, base_link 品红高亮)

    Foxglove Studio 用 "Foxglove WebSocket" 连接 ws://<host>:<port>。
    """

    LEGS = ("LF", "RF", "LB", "RB")

    def __init__(self, m, d, host="127.0.0.1", port=8765, enabled=True):
        self._m = m
        self._d = d
        self._enabled = enabled
        self._server = None
        if not enabled:
            return
        if not _HAS_FOXGLOVE:
            self._enabled = False
            print("[Foxglove] 未安装 foxglove-sdk，已自动禁用 Foxglove 发布（可 `pip install foxglove-sdk` 启用）")
            return

        self._tf = FChan.FrameTransformChannel("/tf")
        self._joints = FChan.JointStatesChannel("/joint_states")
        # 每个关节一个 topic（foxglove.JointState 官方 schema），Plot 面板字段原生可选
        self._joint_chans = {
            name: FChan.JointStateChannel(f"/joints/{name}")
            for name in MJ_JOINT_NAMES
        }
        self._imu_quat = FChan.QuaternionChannel("/imu/orientation")
        self._imu_gyro = FChan.Vector3Channel("/imu/angular_velocity")
        self._imu_acc = FChan.Vector3Channel("/imu/linear_acceleration")
        self._imu_grav = FChan.Vector3Channel("/imu/projected_gravity")
        self._scene = FChan.SceneUpdateChannel("/scene")
        self._contacts = FChan.SceneUpdateChannel("/contacts")

        # 机体 / 足端 / 腿部 body id (用于骨架和 TF)
        self._base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._foot_ids = {
            leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_FOOT_LINK")
            for leg in self.LEGS
        }
        self._leg_ids = {
            leg: [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_HIP_LINK"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_KENN_LINK"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_FOOT_LINK"),
            ]
            for leg in self.LEGS
        }

        try:
            self._server = foxglove.start_server(name="rpwd_sim2sim", host=host, port=port)
            print(f"[Foxglove] 服务已启动: ws://{host}:{port}  (topics: /tf /joint_states /joints/* /imu/* /scene /contacts)")
        except Exception as e:
            self._enabled = False
            print(f"[Foxglove] 启动失败 (端口被占用?): {e}")

    # ── 工具 ──
    @staticmethod
    def _stamp():
        now = time.time_ns()
        return FMsg.Timestamp(now // 1_000_000_000, now % 1_000_000_000)

    @staticmethod
    def _vec3(xyz):
        x, y, z = xyz
        return FMsg.Vector3(x=float(x), y=float(y), z=float(z))

    @staticmethod
    def _quat_xyzw(wxyz):
        # MuJoCo xquat / imu_quat 是 [w,x,y,z] → Foxglove Quaternion (x,y,z,w)
        w, x, y, z = wxyz
        return FMsg.Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    @staticmethod
    def _point3(xyz):
        x, y, z = xyz
        return FMsg.Point3(x=float(x), y=float(y), z=float(z))

    @staticmethod
    def _world2self(wxyz, v):
        """把世界系向量 v 旋转到机体坐标系（q* ⊗ v ⊗ q，逆旋转）。wxyz=[w,x,y,z]。"""
        w, x, y, z = wxyz
        vx, vy, vz = v
        scale = 2.0 * w * w - 1.0
        two_w = 2.0 * w
        dot = x * vx + y * vy + z * vz
        cx = y * vz - z * vy   # q_vec × v
        cy = z * vx - x * vz
        cz = x * vy - y * vx
        return (
            scale * vx - two_w * cx + 2.0 * dot * x,
            scale * vy - two_w * cy + 2.0 * dot * y,
            scale * vz - two_w * cz + 2.0 * dot * z,
        )

    # ── 主发布 ──
    def publish(self):
        if not self._enabled:
            return
        d = self._d
        stamp = self._stamp()

        # 1) /joint_states: 16 关节 (MuJoCo 原生顺序)
        joints = [
            FMsg.JointState(
                name=name,
                position=float(d.qpos[7 + i]),
                velocity=float(d.qvel[6 + i]),
                effort=float(d.actuator_force[i]),
            )
            for i, name in enumerate(MJ_JOINT_NAMES)
        ]
        self._joints.log(FMsg.JointStates(timestamp=stamp, joints=joints))

        # 1b) /joints/<NAME>: 每关节一个 topic (foxglove.JointState)，Plot 面板按 topic 名选择
        for i, name in enumerate(MJ_JOINT_NAMES):
            self._joint_chans[name].log(joints[i])

        # 2) /tf: 机身 + 4 足端
        base_pos = self._vec3(d.xpos[self._base_id])
        base_quat = self._quat_xyzw(d.xquat[self._base_id])
        self._tf.log(FMsg.FrameTransform(
            timestamp=stamp, parent_frame_id="world", child_frame_id="base_link",
            translation=base_pos, rotation=base_quat,
        ))
        for leg in self.LEGS:
            fid = self._foot_ids[leg]
            self._tf.log(FMsg.FrameTransform(
                timestamp=stamp, parent_frame_id="world", child_frame_id=f"{leg.lower()}_foot",
                translation=self._vec3(d.xpos[fid]),
                rotation=self._quat_xyzw(d.xquat[fid]),
            ))

        # 3) /imu/*: 姿态/角速度/线加速度（各自独立 topic，官方 schema，字段原生可选）
        imu_q = d.sensor("imu_quat").data   # [w,x,y,z] 机体系→世界系
        imu_g = d.sensor("imu_gyro").data   # gyro 传感器读数
        lin_a = d.qacc[0:3]                 # 机体线加速度 (世界系)
        self._imu_quat.log(FMsg.Quaternion(x=float(imu_q[1]), y=float(imu_q[2]),
                                           z=float(imu_q[3]), w=float(imu_q[0])))
        self._imu_gyro.log(FMsg.Vector3(x=float(imu_g[0]), y=float(imu_g[1]), z=float(imu_g[2])))
        self._imu_acc.log(FMsg.Vector3(x=float(lin_a[0]), y=float(lin_a[1]), z=float(lin_a[2])))
        # 投影重力：世界系 [0,0,-1] 转到机体系，与策略观测 projected_gravity 一致（单位向量）
        pg = self._world2self(imu_q, (0.0, 0.0, -1.0))
        self._imu_grav.log(FMsg.Vector3(x=float(pg[0]), y=float(pg[1]), z=float(pg[2])))

        # 4) /scene: 简易骨架 (base 立方体 + 4 条腿折线)
        lines = []
        for leg in self.LEGS:
            hip, kenn, foot = self._leg_ids[leg]
            lines.append(FMsg.LinePrimitive(
                type=FMsg.LinePrimitiveLineType.LineStrip,
                thickness=0.01, scale_invariant=False,
                pose=FMsg.Pose(position=FMsg.Vector3(x=0.0, y=0.0, z=0.0),
                               orientation=FMsg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                points=[
                    self._point3(d.xpos[self._base_id]),
                    self._point3(d.xpos[hip]),
                    self._point3(d.xpos[kenn]),
                    self._point3(d.xpos[foot]),
                ],
                color=FMsg.Color(r=0.95, g=0.6, b=0.1, a=1.0),
            ))
        entity = FMsg.SceneEntity(
            timestamp=stamp, frame_id="world", id="rpwd_skeleton", frame_locked=False,
            cubes=[FMsg.CubePrimitive(
                pose=FMsg.Pose(position=base_pos, orientation=base_quat),
                size=FMsg.Vector3(x=0.45, y=0.30, z=0.12),
                color=FMsg.Color(r=0.35, g=0.45, b=0.8, a=1.0),
            )],
            lines=lines,
        )
        self._scene.log(FMsg.SceneUpdate(deletions=[], entities=[entity]))

        # 5) /contacts: 接触点球体标记 (排查碰撞/弹跳)
        self._publish_contacts(stamp)

    def _publish_contacts(self, stamp):
        """把 MuJoCo 接触点发布成 /contacts 的球体标记。

        每个接触点一个球: 颜色按接触力绿(小)→红(大), 大小随力增大;
        base_link 参与接触时用亮品红色高亮, 便于定位「机身被弹」的接触源。
        """
        d = self._d
        spheres = []
        for i in range(d.ncon):
            c = d.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self._m, d, i, force)
            fmag = float(np.linalg.norm(force[:3]))
            t = min(1.0, fmag / 100.0)              # 100N 封顶
            diam = 0.02 + 0.04 * t                   # 2~6cm 球
            b1 = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, self._m.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, self._m.geom_bodyid[c.geom2])
            if b1 == "base_link" or b2 == "base_link":
                r, g, b = 1.0, 0.0, 1.0              # 品红高亮
            else:
                r, g, b = t, 1.0 - t, 0.0            # 绿→红
            spheres.append(FMsg.SpherePrimitive(
                pose=FMsg.Pose(
                    position=FMsg.Vector3(x=float(c.pos[0]), y=float(c.pos[1]), z=float(c.pos[2])),
                    orientation=FMsg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                size=FMsg.Vector3(x=diam, y=diam, z=diam),
                color=FMsg.Color(r=r, g=g, b=b, a=0.85),
            ))
        self._contacts.log(FMsg.SceneUpdate(
            deletions=[],
            entities=[FMsg.SceneEntity(
                timestamp=stamp, frame_id="world", id="rpwd_contacts", frame_locked=False,
                spheres=spheres,
            )],
        ))
