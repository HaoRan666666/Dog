"""RP_wd sim2sim: Isaac Lab 策略 → MuJoCo 部署"""

import torch
import torch.nn as nn
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

# Isaac Lab 观测/动作关节顺序 (按类型分组: ABAD→HIP→KENN→FOOT)
# ABAD: LF, RF, LB, RB  →  mj 0,4,8,12
# HIP:  LF, RF, LB, RB  →  mj 1,5,9,13
# KENN: LF, RF, LB, RB  →  mj 2,6,10,14
# FOOT: LF, RF, LB, RB  →  mj 3,7,11,15

# 12个腿部关节 (不含轮子) 在 Isaac Lab 中的顺序 → MuJoCo qpos 索引
MJ_POS_IDX_FOR_LAB_LEG = [0, 4, 8, 12,   # ABAD
                           1, 5, 9, 13,   # HIP
                           2, 6, 10, 14]  # KENN

# 全部16个关节在 Isaac Lab 中的顺序 → MuJoCo qvel 索引
MJ_VEL_IDX_FOR_LAB = [0, 4, 8, 12,       # ABAD
                       1, 5, 9, 13,       # HIP
                       2, 6, 10, 14,      # KENN
                       3, 7, 11, 15]      # FOOT

# Isaac Lab action → MuJoCo ctrl 索引
LAB_ACT_TO_MJ_CTRL = [0, 4, 8, 12,       # ABAD
                       1, 5, 9, 13,       # HIP
                       2, 6, 10, 14,      # KENN
                       3, 7, 11, 15]      # FOOT

# ── 默认关节角度 (Isaac Lab 顺序) ────────────────────────────────────
DEFAULT_DOF_POS = torch.tensor([
    0.0, 0.0, 0.0, 0.0,        # ABAD
    0.72, 0.72, 0.72, 0.72,    # HIP (Isaac Lab init_state)
    -1.41, -1.41, -1.41, -1.41, # KENN (Isaac Lab init_state)
    0.0, 0.0, 0.0, 0.0,         # FOOT
], device=device, dtype=torch.float32)

# ── Action scale (与 walk_env_cfg.py 一致) ────────────────────────────
ACTIONS_SCALE = torch.tensor([
    0.125, 0.125, 0.125, 0.125,   # ABAD
    0.25, 0.25, 0.25, 0.25,       # HIP
    0.25, 0.25, 0.25, 0.25,       # KENN
    5.0, 5.0, 5.0, 5.0,           # FOOT (velocity)
], device=device, dtype=torch.float32)

# ── 观测归一化 (从 checkpoint 加载) ──────────────────────────────────
obs_mean = torch.zeros(159, device=device)
obs_std = torch.ones(159, device=device)


def world2self(quat, v):
    """将世界坐标系向量旋转到机体坐标系.
    quat: [qw, qx, qy, qz] (Isaac Lab/MuJoCo 均为 wxyz 标量在前)
    v: [x, y, z]
    公式: v' = q* ⊗ v ⊗ q (逆旋转), 展开:
    v' = v*(2qw²-1) - 2qw*(q_vec×v) + 2(q_vec·v)*q_vec
    即: a - b + c (其中 a=标量项, b=叉乘项, c=点乘项)
    """
    q_w = quat[0]
    q_vec = quat[1:]
    v_vec = v.to(device=device, dtype=torch.float32)
    a = v_vec * (2.0 * q_w ** 2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    return a - b + c


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

    Isaac Lab 的 concatenate_terms=True 拼接顺序:
    每个观测 term 先堆叠自己的 3 帧历史, 再按 term 顺序拼接。
    即: [T0_f0, T0_f1, T0_f2, T1_f0, T1_f1, T1_f2, T2_f0, ...]
    而不是按帧拼接: [T0_f0, T1_f0, T2_f0, ..., T0_f1, T1_f1, ...]

    history: deque of up to 3 single-frame tensors (oldest first).
    每个 frame 53 维: ang_vel(3) + grav(3) + cmd(3) + pos(12) + vel(16) + act(16)
    """
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
    obs = torch.cat(parts)

    obs = (obs - obs_mean) / obs_std.clamp(min=1e-6)
    return obs


def load_policy(ckpt_path):
    """从 RSL-RL checkpoint 中提取 actor 网络。"""
    global obs_mean, obs_std

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # 提取 actor 权重 (去除 "actor." 前缀)
    actor_state = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state[k[len("actor."):]] = v

    # 提取观测归一化参数 (RSL-RL key: actor_obs_normalizer._mean/_std)
    if "actor_obs_normalizer._mean" in state_dict:
        obs_mean = state_dict["actor_obs_normalizer._mean"].squeeze(0).to(device)
        obs_std = state_dict["actor_obs_normalizer._std"].squeeze(0).to(device)
        print(f"已加载观测归一化参数 (mean shape={obs_mean.shape})")

    # 构建网络: 159 → 512 → 256 → 128 → 16
    actor = nn.Sequential(
        nn.Linear(159, 512),
        nn.ELU(alpha=1.0),
        nn.Linear(512, 256),
        nn.ELU(alpha=1.0),
        nn.Linear(256, 128),
        nn.ELU(alpha=1.0),
        nn.Linear(128, 16),
    ).to(device)

    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def main():
    from collections import deque

    # ── 加载策略 ──────────────────────────────────────────────────
    ckpt_path = "/home/rp/model_server/model_1000.pt"
    try:
        actor = load_policy(ckpt_path)
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

    actions = torch.zeros(16, device=device, dtype=torch.float32)
    history = deque(maxlen=3)
    print("策略启动")

    # ── 启动 MuJoCo 渲染 ──────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:

        while viewer.is_running():
            commands = (0.0, 0.0, .0)

            single_obs = get_single_obs(actions=actions, commands=commands)
            history.append(single_obs)
            obs = build_history_obs(history)
            obs = torch.clip(obs, -100, 100)

            with torch.no_grad():
                actions = actor(obs)

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
