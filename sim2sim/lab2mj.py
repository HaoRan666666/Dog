import argparse
import torch
import mujoco
import mujoco.viewer
import time
import os

# ── 加载 MuJoCo 模型 ────────────────────────────────────────────────
xml_path = os.path.join(os.path.dirname(__file__),
                        "../source/Dog/Dog/assets/BPX/mujoco/bpx.xml")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 关节名称 ──────────────────────────────────────────────────────
# Isaac Lab 中 `find_joints([".*"], preserve_order=False)` 的结果遵循
# articulation 的自然 DOF 顺序（per-leg），与 MuJoCo actuator 顺序一致。
joint_names = [
    "fl_hip_roll_joint", "fl_hip_pitch_joint", "fl_knee_joint",
    "fr_hip_roll_joint", "fr_hip_pitch_joint", "fr_knee_joint",
    "hl_hip_roll_joint", "hl_hip_pitch_joint", "hl_knee_joint",
    "hr_hip_roll_joint", "hr_hip_pitch_joint", "hr_knee_joint",
]


# ── 传感器读取 ──────────────────────────────────────────────────────
def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx: start_idx + dim]
    return torch.tensor(sensor_values, device=device, dtype=torch.float32)


def set_joint_angle(joint_name, angle):
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qpos[m.jnt_qposadr[joint_id]] = angle


# ── 四元数旋转（世界系 → 机体系）────────────────────────────────────
def world2self(quat, v):
    q_w = quat[0]
    q_vec = quat[1:]
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, device=device, dtype=torch.float32)
    v_vec = v.clone().detach().to(device=device, dtype=torch.float32)
    a = v_vec * (2.0 * q_w ** 2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)


# ── 构建观测 ────────────────────────────────────────────────────────
def get_obs(actions, default_dof_pos, commands=None):
    if commands is None:
        commands = [1.0, 0.0, 0.0]
    commands_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    # imu 角速度（机体角速度）
    imu_gyro = get_sensor_data("body_gyro")
    # 投影重力方向
    base_quat = get_sensor_data("body_quat")
    projected_gravity = world2self(base_quat,
                                   torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32))
    # 关节位置与速度（Isaac Lab 顺序）
    dof_pos = torch.zeros(12, device=device, dtype=torch.float32)
    dof_vel = torch.zeros(12, device=device, dtype=torch.float32)
    for i in range(12):
        dof_pos[i] = get_sensor_data(joint_names[i] + "_pos")[0]
        dof_vel[i] = get_sensor_data(joint_names[i] + "_vel")[0]

    cmds = torch.tensor(commands, device=device, dtype=torch.float32)

    return torch.cat(
        [
            imu_gyro,                      # 3  训练时 scale=null，使用原始值
            projected_gravity,             # 3  训练时 scale=null
            cmds * commands_scale,         # 3
            (dof_pos - default_dof_pos),   # 12 训练时 scale=null
            dof_vel,                       # 12 训练时 scale=null
            actions,                       # 12
        ],
        axis=-1,
    )


# ── 主循环 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BPX 机器人 sim2sim 迁移")
    parser.add_argument("--policy", type=str, required=True,
                        help="策略文件路径 (.pt)")
    args = parser.parse_args()

    loaded_policy = torch.jit.load(args.policy)
    loaded_policy.eval()
    loaded_policy.to(device)
    print(f"策略加载成功: {args.policy}")

    # articulation 的 default_joint_pos = init_state.joint_pos（站立姿态）
    # joint_pos_rel = joint_pos - standing_pose（相对于站立）
    # action offset = default_joint_pos = standing_pose
    default_dof_pos = torch.tensor([
        0.0, 0.8, -1.5,   # fl: hip_roll, hip_pitch, knee
        0.0, 0.8, -1.5,   # fr
        0.0, 1.0, -1.5,   # hl
        0.0, 1.0, -1.5,   # hr
    ], device=device, dtype=torch.float32)

    # 动作缩放（来自 ActionsCfg: scale=0.25，所有关节统一）
    actions_scale = torch.full((12,), 0.25, device=device, dtype=torch.float32)

    actions = torch.zeros(12, device=device, dtype=torch.float32)

    # 对齐 Isaac Lab 控制频率: dt=1/120, decimation=2 → 控制周期=1/60≈0.01667s
    # MuJoCo timestep=0.002, 0.01667/0.002≈8 步/控制周期
    decimation = 8

    # ── 初始化机器人到站立姿态 ─────────────────────────────────────
    # 设置基座高度 → Isaac Lab init_state pos z=0.42 (匹配 XML)
    base_freejoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
    base_qpos_adr = m.jnt_qposadr[base_freejoint_id]
    d.qpos[base_qpos_adr:base_qpos_adr + 7] = [0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0]
    # 设置关节到站立角度 + 施加 PD 控制保持姿态
    standing_ctrl = default_dof_pos.cpu().numpy()
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[joint_id]] = standing_ctrl[i]
        d.ctrl[i] = standing_ctrl[i]
    # 稳定物理（PD 控制保持站立）
    for _ in range(500):
        mujoco.mj_step(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # 构建观测（关节顺序与 MuJoCo actuator 一致）
            obs = get_obs(actions=actions, default_dof_pos=default_dof_pos)
            obs = torch.clip(obs, -100, 100)

            # 策略推理
            actions = loaded_policy(obs)

            # 计算关节目标位置: raw_action * scale + default_joint_pos
            act = actions * actions_scale + default_dof_pos
            act = torch.clip(act, -100, 100)
            act = act.detach().cpu().numpy()
            for i in range(12):
                d.ctrl[i] = act[i]

            step_start = time.time()
            for _ in range(decimation):
                mujoco.mj_step(m, d)

            viewer.sync()

            time_until_next_step = m.opt.timestep * decimation - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
