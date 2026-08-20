"""Foxglove 实时数据发布桥：把 MuJoCo 状态发布成 Foxglove topic。

独立于 sim2sim 部署脚本，供 lab2mujoco2.py 等入口按需调用：
    bridge = FoxgloveBridge(m, d, host=..., port=..., enabled=...)
    bridge.publish()
"""

import time

import foxglove
import mujoco
from foxglove import channels as FChan
from foxglove import messages as FMsg

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
      /joint_states foxglove.JointStates     16 关节位置/速度/力矩 (Plot 面板)
      /imu          json                     姿态四元数 + 角速度 + 线加速度
      /scene        foxglove.SceneUpdate     简易骨架 (base 立方体 + 4 条腿折线)

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

        self._tf = FChan.FrameTransformChannel("/tf")
        self._joints = FChan.JointStatesChannel("/joint_states")
        self._imu = foxglove.Channel("/imu", message_encoding="json")
        self._scene = FChan.SceneUpdateChannel("/scene")

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
            print(f"[Foxglove] 服务已启动: ws://{host}:{port}  (topics: /tf /joint_states /imu /scene)")
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

        # 3) /imu (json): 姿态 + 角速度 + 线加速度
        imu_q = d.sensor("imu_quat").data   # [w,x,y,z] 机体系→世界系
        imu_g = d.sensor("imu_gyro").data   # gyro 传感器读数
        lin_a = d.qacc[0:3]                 # 机体线加速度 (世界系)
        self._imu.log({
            "orientation": {"w": float(imu_q[0]), "x": float(imu_q[1]),
                            "y": float(imu_q[2]), "z": float(imu_q[3])},
            "angular_velocity": {"x": float(imu_g[0]), "y": float(imu_g[1]), "z": float(imu_g[2])},
            "linear_acceleration": {"x": float(lin_a[0]), "y": float(lin_a[1]), "z": float(lin_a[2])},
        })

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
