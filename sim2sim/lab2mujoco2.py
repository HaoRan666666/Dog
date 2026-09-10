"""RP_wd sim2sim: Isaac Lab 策略 → MuJoCo 部署"""

import argparse
import torch
import mujoco
import mujoco.viewer
import time
import os

from foxglove_bridge import FoxgloveBridge
from gamepad_simple import GamepadSimple

# ── 加载 MuJoCo 模型 ─────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__),
                        "../source/Dog/Dog/assets/RP_wd/mjcf/RP_wd.xml")
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # gyro 输出的是 IMU site 局部坐标系(机体坐标系)下的角速度，可直接作为 base_ang_vel
    base_quat = torch.tensor(
        d.sensor('imu_quat').data.copy(),
        device=device,
        dtype=torch.float32
    )

    gyro_local = torch.tensor(
        d.sensor('imu_gyro').data.copy(),
        device=device,
        dtype=torch.float32
    )

    # 如果 IMU site 与 base_link 坐标轴一致，可以直接作为 base_ang_vel
    base_ang_vel = gyro_local
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


def main():
    from collections import deque

    parser = argparse.ArgumentParser(description="Isaac Lab→MuJoCo sim2sim 部署")
    parser.add_argument("--ckpt", type=str,
                        default="/home/rp/dog/Dog/logs/rsl_rl/RP_wd_walk_flat/2026-09-08_20-31-40/exported/policy.pt",
                        help="平地模型 (默认模型) TorchScript 策略路径")
    parser.add_argument("--biped-ckpt", type=str,
                        default="/home/rp/rpwd_model/rpwd_biped_policy/policy.pt",
                        help="双轮站立模型 TorchScript 策略路径 (手柄 Y 键切换)")
    parser.add_argument("--platform-ckpt", type=str,
                        default="/home/rp/model_server/rpwd_platform_save_success/exported/policy.pt",
                        help="高台模型 TorchScript 策略路径 (手柄 Y 键切换)")
    parser.add_argument("--step-ckpt", type=str,
                        default="/home/rp/model_server/rpwd_step_success/exported/policy.pt",
                        help="台阶模型 TorchScript 策略路径 (手柄 Y 键切换)")
    parser.add_argument("--input", type=str, default="keyboard", choices=["keyboard", "gamepad"],
                        help="输入设备: keyboard (零指令站立) 或 gamepad")
    parser.add_argument("--device", type=str, default="/dev/input/js0",
                        help="手柄设备路径")
    parser.add_argument("--y-button", type=int, default=3,
                        help="手柄 Y 键在 /dev/input/jsX 中的按键编号 (若切换无效，参考运行时打印的按键编号校准)")
    parser.add_argument("--vx", type=float, default=2.0, help="前进灵敏度")
    parser.add_argument("--vy", type=float, default=2.0, help="横移灵敏度")
    parser.add_argument("--wz", type=float, default=2.0, help="转向灵敏度")
    parser.add_argument("--foxglove-host", type=str, default="0.0.0.0",
                        help="Foxglove WebSocket 绑定地址")
    parser.add_argument("--foxglove-port", type=int, default=8765,
                        help="Foxglove WebSocket 端口")
    parser.add_argument("--no-foxglove", action="store_true",
                        help="关闭 Foxglove 数据发布")
    args = parser.parse_args()

    # ── 加载策略 (TorchScript 模型，已内置归一化) ─────────────────
    # 所有模型都提前加载好，手柄 Y 键切换时直接换用已加载的模块，无需重新读盘。
    # 切换顺序按此列表循环：flat → biped → platform → step → flat → ...
    MODEL_ORDER = ["flat", "biped", "platform", "step"]
    model_paths = {
        "flat": args.ckpt,
        "biped": args.biped_ckpt,
        "platform": args.platform_ckpt,
        "step": args.step_ckpt,
    }
    policies = {}
    for name in MODEL_ORDER:
        path = model_paths[name]
        try:
            p = torch.jit.load(path)
            p.eval()
            p.to(device)
            policies[name] = p
            print(f"策略加载成功 [{name}]: {path}")
        except Exception as e:
            policies[name] = None
            print(f"策略加载失败 [{name}] ({path}): {e}")

    if policies["flat"] is None:
        print("默认模型 [flat] 加载失败，无法启动")
        exit()

    active_model = "flat"
    policy = policies[active_model]

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
        print("手柄控制: 左摇杆移动, 右摇杆左右转向, Y 键循环切换 平地/双轮/高台/台阶 模型")
    else:
        input_device = None
        print("键盘模式: 零指令站立 (机器人原地维持平衡)")

    actions = torch.zeros(16, device=device, dtype=torch.float32)
    history = deque(maxlen=3)
    prev_y_state = 0
    print("策略启动")

    # ── Foxglove 数据发布 ─────────────────────────────────────────
    foxglove_bridge = FoxgloveBridge(m, d, host=args.foxglove_host,
                                     port=args.foxglove_port,
                                     enabled=not args.no_foxglove)

    # ── 启动 MuJoCo 渲染 ──────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 相机跟随机器人 (base_link)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.lookat = [0, 0, 0.3]   # 视线中心: 机身略上方
        viewer.cam.distance = 3.0         # 相机距离 (m)
        viewer.cam.azimuth = 0            # 水平方位角: 0=正后方, 90/270=两侧, 180=正前方
        viewer.cam.elevation = -20        # 俯仰角: 越负越俯视(越高), 0=平视, 正值会钻地下

        while viewer.is_running():
            if input_device is not None:
                raw = input_device.advance()

                # Y 键边沿触发：按 MODEL_ORDER 循环切换模型，跳过加载失败的
                y_state = 0
                if 0 <= args.y_button < len(input_device.buttons):
                    y_state = input_device.buttons[args.y_button]
                if y_state == 1 and prev_y_state == 0:
                    cur_idx = MODEL_ORDER.index(active_model)
                    for step in range(1, len(MODEL_ORDER) + 1):
                        candidate = MODEL_ORDER[(cur_idx + step) % len(MODEL_ORDER)]
                        if policies[candidate] is not None:
                            active_model = candidate
                            policy = policies[active_model]
                            # 各模型的观测历史/动作含义不同，切换时清空避免污染
                            actions = torch.zeros(16, device=device, dtype=torch.float32)
                            history.clear()
                            print(f"切换模型 → {active_model}")
                            break
                    else:
                        print("切换失败：没有其他已成功加载的模型")
                prev_y_state = y_state

                # 训练约定: +wz=逆时针 (右手定则), 手柄摇杆原始输出已取反 → 直接使用
                commands = (raw[0], raw[1], raw[2])
                if active_model == "biped":
                    # 双轮任务训练时 lin_vel_y 指令范围固定为 (0,0)，部署时同步禁用左右横移
                    commands = (commands[0], 0.0, commands[2])
            else:
                commands = (0.0, 0.0, 0.0)

            single_obs = get_single_obs(actions=actions, commands=commands)
            history.append(single_obs)
            obs = build_history_obs(history)
            obs = torch.clip(obs, -150, 150)

            with torch.no_grad():
                actions = policy(obs.unsqueeze(0)).squeeze(0)

            # action → actuator ctrl
            act = actions * ACTIONS_SCALE + DEFAULT_DOF_POS
            act = torch.clip(act, -150, 150)
            act_np = act.detach().cpu().numpy()

            for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
                d.ctrl[mj_idx] = act_np[lab_idx]

            # 8 步物理仿真 = 0.02s = 50Hz
            step_start = time.time()
            for _ in range(steps_per_control):
                mujoco.mj_step(m, d)

            foxglove_bridge.publish()

            viewer.sync()
            time_until_next = control_dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
