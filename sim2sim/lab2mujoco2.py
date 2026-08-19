"""RP_wd sim2sim: Isaac Lab 策略 → MuJoCo 部署"""

import argparse
import torch
import mujoco
import mujoco.viewer
import time
import os

# ── 加载 MuJoCo 模型 ─────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__),
                        "../source/Dog/Dog/assets/RP_wd/mjcf/RP_wd.xml")
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Isaac Lab 真实关节顺序 (按类型分组: ABAD→HIP→KENN→FOOT, 每组内 LB→LF→RB→RF)
# MuJoCo 原生顺序: LF→RF→LB→RB × ABAD→HIP→KENN→FOOT

# 12个腿部关节 (不含轮子) 在 Isaac Lab 中的顺序 → MuJoCo qpos 索引
MJ_POS_IDX_FOR_LAB_LEG = [8, 0, 12, 4,   # ABAD: LB→mj8, LF→mj0, RB→mj12, RF→mj4
                           9, 1, 13, 5,   # HIP:  LB→mj9, LF→mj1, RB→mj13, RF→mj5
                          10, 2, 14, 6]   # KENN: LB→mj10, LF→mj2, RB→mj14, RF→mj6

# 全部16个关节在 Isaac Lab 中的顺序 → MuJoCo qvel 索引
MJ_VEL_IDX_FOR_LAB = [8, 0, 12, 4,       # ABAD: LB→mj8, LF→mj0, RB→mj12, RF→mj4
                       9, 1, 13, 5,       # HIP:  LB→mj9, LF→mj1, RB→mj13, RF→mj5
                      10, 2, 14, 6,       # KENN: LB→mj10, LF→mj2, RB→mj14, RF→mj6
                      11, 3, 15, 7]       # FOOT: LB→mj11, LF→mj3, RB→mj15, RF→mj7

# Isaac Lab action → MuJoCo ctrl 索引
LAB_ACT_TO_MJ_CTRL = [8, 0, 12, 4,       # ABAD: LB→mj8, LF→mj0, RB→mj12, RF→mj4
                       9, 1, 13, 5,       # HIP:  LB→mj9, LF→mj1, RB→mj13, RF→mj5
                      10, 2, 14, 6,       # KENN: LB→mj10, LF→mj2, RB→mj14, RF→mj6
                      11, 3, 15, 7]       # FOOT: LB→mj11, LF→mj3, RB→mj15, RF→mj7

# ── 默认关节角度 (Isaac Lab 顺序: LB→LF→RB→RF) ──────────────────────
DEFAULT_DOF_POS = torch.tensor([
    0.0, 0.0, 0.0, 0.0,              # ABAD: LB, LF, RB, RF
    0.72, 0.72, 0.72, 0.72,          # HIP:  LB, LF, RB, RF
    -1.41, -1.41, -1.41, -1.41,      # KENN: LB, LF, RB, RF
    0.0, 0.0, 0.0, 0.0,              # FOOT: LB, LF, RB, RF
], device=device, dtype=torch.float32)

# ── Action scale (Isaac Lab 顺序: LB→LF→RB→RF) ──────────────────────
ACTIONS_SCALE = torch.tensor([
    0.125, 0.125, 0.125, 0.125,      # ABAD: LB, LF, RB, RF
    0.25, 0.25, 0.25, 0.25,          # HIP:  LB, LF, RB, RF
    0.25, 0.25, 0.25, 0.25,          # KENN: LB, LF, RB, RF
    5.0, 5.0, 5.0, 5.0,              # FOOT: LB, LF, RB, RF
], device=device, dtype=torch.float32)

# def world2self(quat, v):
#     """将世界坐标系向量旋转到机体坐标系.
#     quat: [qw, qx, qy, qz] (Isaac Lab/MuJoCo 均为 wxyz 标量在前)
#     v: [x, y, z]
#     公式: v' = q* ⊗ v ⊗ q (逆旋转), 展开:
#     v' = v*(2qw²-1) - 2qw*(q_vec×v) + 2(q_vec·v)*q_vec
#     即: a - b + c (其中 a=标量项, b=叉乘项, c=点乘项)
#     """
#     q_w = quat[0]
#     q_vec = quat[1:]
#     v_vec = v.to(device=device, dtype=torch.float32)
#     a = v_vec * (2.0 * q_w ** 2 - 1.0)
#     b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
#     c = q_vec * torch.dot(q_vec, v_vec) * 2.0
#     return a - b + c

def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device, dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)

def get_single_obs(actions, commands=(0.0, 0.0, 0.0)):
    """构建单帧观测 (53维)，不含历史堆叠。"""
    # IMU 传感器：framequat [qw,qx,qy,qz] (机体系→世界系)
    # gyro 返回的是世界坐标系角速度，需要转到机体坐标系
    base_quat = torch.tensor(d.sensor('imu_quat').data.copy(), device=device, dtype=torch.float32)
    gyro_world = torch.tensor(d.sensor('imu_gyro').data.copy(), device=device, dtype=torch.float32)
    base_ang_vel = world2self(base_quat, gyro_world)

    # 投影重力: world→body
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
    projected_gravity = world2self(base_quat, gravity)

    # 指令
    cmds = torch.tensor(commands, device=device, dtype=torch.float32)

    # 关节位置 (仅腿部 12 个)
    dof_pos = torch.tensor(d.qpos[7:].copy(), device=device, dtype=torch.float32)
    dof_pos_lab = dof_pos[MJ_POS_IDX_FOR_LAB_LEG]

    # 关节速度 (全部 16 个)
    dof_vel = torch.tensor(d.qvel[6:].copy(), device=device, dtype=torch.float32)
    dof_vel_lab = dof_vel[MJ_VEL_IDX_FOR_LAB]

    return torch.cat([
        base_ang_vel ,                      # 3
        projected_gravity,                         # 3
        cmds,                                      # 3
        (dof_pos_lab - DEFAULT_DOF_POS[:12]),      # 12
        dof_vel_lab ,                        # 16
        actions,                                   # 16
    ])


def build_history_obs(history):
    """堆叠 3 帧历史 → 159 维观测并归一化。

    Isaac Lab 的 concatenate_terms=True + flatten_history_dim=True 拼接顺序:
    每个观测 term 先堆叠自己的 3 帧历史, 再按 term 顺序拼接。
    即: [term0_f0, term0_f1, term0_f2, term1_f0, term1_f1, term1_f2, ...]"""

    # 零填充
    while len(history) < 3:
        history.appendleft(torch.zeros(53, device=device))

    # 6 个 term 各自的维度和在单帧中的起止索引
    term_slices = [
        (0, 3),    # base_ang_vel
        (3, 6),    # projected_gravity
        (6, 9),    # velocity_commands
        (9, 21),   # joint_pos (12)
        (21, 37),  # joint_vel (16)
        (37, 53),  # actions (16)
    ]

    parts = []
    for start, end in term_slices:
        for h in history:
            parts.append(h[start:end])
    return torch.cat(parts)


# ═══════════════════════════════════════════════════════════════════════
# 游戏手柄输入设备
# ═══════════════════════════════════════════════════════════════════════

class GamepadSimple:
    """直接读取 Linux /dev/input/jsX 设备文件，输出 SE(2) 速度指令。

    绕过 Omniverse 的手柄接口，直接从内核驱动层读取。
    """

    def __init__(self, vx=1.0, vy=1.0, wz=2.0, dead_zone=0.1, dev="/dev/input/js0"):
        self._vx = vx
        self._vy = vy
        self._wz = wz
        self._dead_zone = dead_zone
        self._fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        self._axes = [0.0] * 8
        self._buttons = [0] * 16
        print(f"手柄已连接: {dev}")

    def _poll(self):
        """非阻塞读取手柄事件，更新轴和按键状态。"""
        import struct
        try:
            while True:
                data = os.read(self._fd, 8)
                if not data:
                    break
                _time, value, ev_type, number = struct.unpack('IhBB', data)
                ev_type &= 0x7f
                if ev_type == 2:                          # 轴事件
                    self._axes[number] = value / 32767.0  # 归一化到 [-1, 1]
                elif ev_type == 1:                        # 按键事件
                    self._buttons[number] = value
        except BlockingIOError:
            pass

    def advance(self) -> tuple:
        """每帧调用，返回 (vx, vy, wz) 速度指令。"""
        self._poll()
        cmd = [0.0, 0.0, 0.0]
        ly = self._axes[1]    # 左摇杆 Y 轴 (向上为负)
        lx = self._axes[0]    # 左摇杆 X 轴
        rx = self._axes[3]    # 右摇杆 X 轴（转向）
        if abs(ly) > self._dead_zone:
            cmd[0] = -ly * self._vx
        if abs(lx) > self._dead_zone:
            cmd[1] = -lx * self._vy
        if abs(rx) > self._dead_zone:
            cmd[2] = -rx * self._wz
        return tuple(cmd)


def main():
    from collections import deque

    parser = argparse.ArgumentParser(description="Isaac Lab→MuJoCo sim2sim 部署")
    parser.add_argument("--ckpt", type=str, default="/home/rp/model_server/rpwd_0816_platform/exported/policy.pt",
                        help="TorchScript 策略路径 (由 play.py 自动导出到 exported/policy.pt)")
    parser.add_argument("--input", type=str, default="keyboard", choices=["keyboard", "gamepad"],
                        help="输入设备: keyboard (零指令站立) 或 gamepad")
    parser.add_argument("--device", type=str, default="/dev/input/js0",
                        help="手柄设备路径")
    parser.add_argument("--vx", type=float, default=1.0, help="前进灵敏度")
    parser.add_argument("--vy", type=float, default=1.0, help="横移灵敏度")
    parser.add_argument("--wz", type=float, default=1.0, help="转向灵敏度")
    args = parser.parse_args()

    # ── 加载策略 (TorchScript 模型，已内置归一化) ─────────────────
    ckpt_path = args.ckpt
    try:
        policy = torch.jit.load(ckpt_path)
        policy.eval()
        policy.to(device)
        print(f"策略加载成功: {ckpt_path}")
    except Exception as e:
        print(f"策略加载失败: {e}")
        exit()

    # ── 控制频率: dt=0.0025, decimation=8 → 50Hz (0.02s) ──
    sim_dt = m.opt.timestep
    decimation = 8
    control_dt = sim_dt * decimation
    steps_per_control = decimation
    print(f"MuJoCo dt={sim_dt:.4f}s, decimation={decimation} → 控制间隔={control_dt}s ({1/control_dt:.0f}Hz)")

    # ── 设置初始姿态到默认站立位 ───────────────────────────────────
    mj_default_qpos = torch.zeros(16, device=device)
    for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
        mj_default_qpos[mj_idx] = DEFAULT_DOF_POS[lab_idx]
    d.qpos[7:] = mj_default_qpos.cpu().numpy()
    d.qpos[2] = 0.5

    # ── 稳定阶段：PD 伺服驱动到默认姿态，等机器人落地 ─────────────
    print("稳定中...")
    for _ in range(500):
        for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
            d.ctrl[mj_idx] = DEFAULT_DOF_POS[lab_idx].item()
        mujoco.mj_step(m, d)

    # ── 输入设备 ──────────────────────────────────────────────────
    if args.input == "gamepad":
        input_device = GamepadSimple(vx=args.vx, vy=args.vy, wz=args.wz, dev=args.device)
        print("手柄控制: 左摇杆移动, 右摇杆左右转向")
    else:
        input_device = None
        print("键盘模式: 零指令站立 (机器人原地维持平衡)")

    actions = torch.zeros(16, device=device, dtype=torch.float32)
    history = deque(maxlen=3)
    print("策略启动")

    # ── 启动 MuJoCo 渲染 ──────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:

        while viewer.is_running():
            if input_device is not None:
                raw = input_device.advance()
                # 训练约定: +wz=逆时针 (右手定则), 手柄摇杆原始输出已取反 → 直接使用
                commands = (raw[0], raw[1], raw[2])
            else:
                commands = (0.0, 0.0, 0.0)

            single_obs = get_single_obs(actions=actions, commands=commands)
            history.append(single_obs)
            obs = build_history_obs(history)
            obs = torch.clip(obs, -100, 100)

            with torch.no_grad():
                actions = policy(obs.unsqueeze(0)).squeeze(0)

            # action → actuator ctrl
            act = actions * ACTIONS_SCALE + DEFAULT_DOF_POS
            act = torch.clip(act, -100, 100)
            act_np = act.detach().cpu().numpy()

            for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
                d.ctrl[mj_idx] = act_np[lab_idx]

            # 8 步物理仿真 = 0.02s = 50Hz
            step_start = time.time()
            for _ in range(steps_per_control):
                mujoco.mj_step(m, d)

            viewer.sync()
            time_until_next = control_dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
