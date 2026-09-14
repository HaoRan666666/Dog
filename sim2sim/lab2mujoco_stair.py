"""RP_wd 台阶任务 sim2sim：深度相机 + EncoderActorCritic 策略 → MuJoCo 部署。

标量观测部分（关节映射、159 维历史拼接）直接照抄 ``lab2mujoco2.py`` 里已经跑通
的写法；新增的是深度相机（``raycaster.DepthCamera``）和 37 帧深度历史采样
（对应训练 ``StairSceneCfg.depth_camera`` 的 ``data_histories=37`` +
``delayed_visualizable_image`` 的 ``history_skip_frames=5, num_output_frames=8``）。
"""

import argparse
import os
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch

# 见 lab2mujoco2.py 里的记录：CPU 上小 MLP/CNN 用 PyTorch 默认多线程反而慢 600 倍
# （线程调度开销远大于计算量），必须单线程。
torch.set_num_threads(1)

from encoder_policy import load_stair_policy
from foxglove_bridge import FoxgloveBridge
from gamepad_simple import GamepadSimple
from raycaster import DepthCamera

# ── 加载 MuJoCo 模型 ─────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__),
                        "../source/Dog/Dog/assets/RP_wd/mjcf/RP_wd.xml")
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 关节映射常量 (照抄 lab2mujoco2.py，含义见该文件注释) ────────────
MJ_POS_IDX_FOR_LAB_LEG = [8, 0, 12, 4,
                           9, 1, 13, 5,
                          10, 2, 14, 6]

MJ_VEL_IDX_FOR_LAB = [8, 0, 12, 4,
                       9, 1, 13, 5,
                      10, 2, 14, 6,
                      11, 3, 15, 7]

LAB_ACT_TO_MJ_CTRL = [8, 0, 12, 4,
                       9, 1, 13, 5,
                      10, 2, 14, 6,
                      11, 3, 15, 7]

DEFAULT_DOF_POS = torch.tensor([
    0.0, 0.0, 0.0, 0.0,
    0.72, 0.72, 0.72, 0.72,
    -1.41, -1.41, -1.41, -1.41,
    0.0, 0.0, 0.0, 0.0,
], device=device, dtype=torch.float32)

ACTIONS_SCALE = torch.tensor([
    0.125, 0.125, 0.125, 0.125,
    0.25, 0.25, 0.25, 0.25,
    0.25, 0.25, 0.25, 0.25,
    5.0, 5.0, 5.0, 5.0,
], device=device, dtype=torch.float32)

# ── 深度历史采样：对应训练 frame_offset = flip([0,5,...,35]) = [35,...,5,0] ──
# (channel 0 = 35 步前最旧的一帧, channel 7 = 最新一帧；见
# Dog/tasks/manager_based/RP_wd/mdp/observations/exteroception.py 里
# delayed_visualizable_image 的 frame_offset 构造逻辑)
DEPTH_HISTORY_LEN = 37
FRAME_OFFSETS = [35, 30, 25, 20, 15, 10, 5, 0]
DEPTH_RANGE = (0.0, 2.0)  # 对应训练 DepthNormalizationCfg(depth_range=(0.0,2.0))


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
    """构建单帧标量观测 (53维)，不含历史堆叠。与 lab2mujoco2.py 完全一致。"""
    base_quat = torch.tensor(d.sensor('imu_quat').data.copy(), device=device, dtype=torch.float32)
    gyro_local = torch.tensor(d.sensor('imu_gyro').data.copy(), device=device, dtype=torch.float32)
    base_ang_vel = gyro_local
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
    projected_gravity = world2self(base_quat, gravity)
    cmds = torch.tensor(commands, device=device, dtype=torch.float32)

    dof_pos = torch.tensor(d.qpos[7:].copy(), device=device, dtype=torch.float32)
    dof_pos_lab = dof_pos[MJ_POS_IDX_FOR_LAB_LEG]

    dof_vel = torch.tensor(d.qvel[6:].copy(), device=device, dtype=torch.float32)
    dof_vel_lab = dof_vel[MJ_VEL_IDX_FOR_LAB]

    return torch.cat([
        base_ang_vel,
        projected_gravity,
        cmds,
        (dof_pos_lab - DEFAULT_DOF_POS[:12]),
        dof_vel_lab,
        actions,
    ])


def build_history_obs(history):
    """堆叠 3 帧标量历史 → 159 维，term-major 顺序，与 lab2mujoco2.py 完全一致。"""
    while len(history) < 3:
        history.appendleft(torch.zeros(53, device=device))

    term_slices = [
        (0, 3), (3, 6), (6, 9), (9, 21), (21, 37), (37, 53),
    ]
    parts = []
    for start, end in term_slices:
        for h in history:
            parts.append(h[start:end])
    return torch.cat(parts)


def build_depth_obs(depth_history):
    """从 37 帧深度历史 deque 里按 FRAME_OFFSETS 采样 8 帧, 归一化, 拼成 (1,8,48,64)。"""
    frames = list(depth_history)
    while len(frames) < DEPTH_HISTORY_LEN:
        frames.insert(0, frames[0])

    near, far = DEPTH_RANGE
    selected = []
    for offset in FRAME_OFFSETS:
        frame = frames[-(offset + 1)]
        normalized = np.clip(frame, near, far) / far
        selected.append(normalized)
    stacked = np.stack(selected, axis=0).astype(np.float32)  # (8, 48, 64)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)  # (1, 8, 48, 64)


_RAY_VIS_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
_IDENTITY_MAT = np.eye(3).flatten()


def draw_ray_vis(viewer, depth_cam, stride):
    """在 viewer.user_scn 里画深度相机射线(红线, 相机→命中点)和命中点(红点)。

    每帧全量重建 user_scn 的 geom 列表(不重设会残留上一帧的线段)。
    """
    cam_pos, hits = depth_cam.get_ray_geometry()
    hits = hits.reshape(depth_cam.v_ray_num, depth_cam.h_ray_num, 3)
    sampled = hits[::stride, ::stride].reshape(-1, 3)
    valid = ~np.isnan(sampled).any(axis=1)
    pts = sampled[valid]

    scn = viewer.user_scn
    ngeom = 0
    for p in pts:
        if ngeom + 2 > scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), _IDENTITY_MAT, _RAY_VIS_RGBA)
        mujoco.mjv_connector(scn.geoms[ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                             2.0, cam_pos, p)
        ngeom += 1
        mujoco.mjv_initGeom(scn.geoms[ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.015, 0.0, 0.0]), p, _IDENTITY_MAT, _RAY_VIS_RGBA)
        ngeom += 1
    scn.ngeom = ngeom


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab→MuJoCo 台阶任务 sim2sim 部署")
    parser.add_argument("--ckpt", type=str,
                        default="/home/rp/model_server/rpwd_0908_stair/model_30000.pt",
                        help="台阶任务原始训练 checkpoint 路径")
    parser.add_argument("--input", type=str, default="keyboard", choices=["keyboard", "gamepad"],
                        help="输入设备: keyboard (零指令站立) 或 gamepad")
    parser.add_argument("--device", type=str, default="/dev/input/js0", help="手柄设备路径")
    parser.add_argument("--vx", type=float, default=2.0, help="前进灵敏度")
    parser.add_argument("--vy", type=float, default=2.0, help="横移灵敏度")
    parser.add_argument("--wz", type=float, default=2.0, help="转向灵敏度")
    parser.add_argument("--debug-vis", action="store_true", help="打开深度相机 cv2 可视化窗口(标定用)")
    parser.add_argument("--ray-vis", action="store_true",
                        help="在 mujoco viewer 3D 场景里画深度相机射线(红线)和命中点(红点)")
    parser.add_argument("--ray-vis-stride", type=int, default=2,
                        help="--ray-vis 降采样步长: 每隔多少个像素画一根射线 (默认2)")
    parser.add_argument("--foxglove-host", type=str, default="0.0.0.0")
    parser.add_argument("--foxglove-port", type=int, default=8765)
    parser.add_argument("--no-foxglove", action="store_true")
    args = parser.parse_args()

    print(f"加载台阶策略: {args.ckpt}")
    policy = load_stair_policy(args.ckpt, device)
    print("策略加载成功")

    depth_cam = DepthCamera(m, d, cam_name="depth_cam", debug_vis=args.debug_vis)

    # ── 控制频率: dt=0.0025, decimation=8 → 50Hz (0.02s), 与相机 update_period 一致 ──
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

    print("稳定中...")
    for _ in range(500):
        for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
            d.ctrl[mj_idx] = DEFAULT_DOF_POS[lab_idx].item()
        mujoco.mj_step(m, d)

    if args.input == "gamepad":
        input_device = GamepadSimple(vx=args.vx, vy=args.vy, wz=args.wz, dev=args.device)
        print("手柄控制: 左摇杆移动, 右摇杆左右转向")
    else:
        input_device = None
        print("键盘模式: 零指令站立 (机器人原地维持平衡)")

    actions = torch.zeros(16, device=device, dtype=torch.float32)
    history = deque(maxlen=3)
    depth_history = deque(maxlen=DEPTH_HISTORY_LEN)
    print("策略启动")

    foxglove_bridge = FoxgloveBridge(m, d, host=args.foxglove_host,
                                     port=args.foxglove_port,
                                     enabled=not args.no_foxglove)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        viewer.cam.lookat = [0, 0, 0.3]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -20

        while viewer.is_running():
            if input_device is not None:
                raw = input_device.advance()
                commands = (raw[0], raw[1], raw[2])
            else:
                commands = (0.0, 0.0, 0.0)

            single_obs = get_single_obs(actions=actions, commands=commands)
            history.append(single_obs)
            scalar_obs = build_history_obs(history)
            scalar_obs = torch.clip(scalar_obs, -150, 150).unsqueeze(0)  # (1, 159)

            depth_frame = depth_cam.step()  # (48, 64), 米
            depth_history.append(depth_frame)
            depth_obs = build_depth_obs(depth_history)  # (1, 8, 48, 64)

            if args.ray_vis:
                draw_ray_vis(viewer, depth_cam, args.ray_vis_stride)

            obs = {"policy": {"obs_flat": scalar_obs, "depth_image": depth_obs}}

            with torch.no_grad():
                actions = policy.act_inference(obs).squeeze(0)

            act = actions * ACTIONS_SCALE + DEFAULT_DOF_POS
            act = torch.clip(act, -150, 150)
            act_np = act.detach().cpu().numpy()

            for lab_idx, mj_idx in enumerate(LAB_ACT_TO_MJ_CTRL):
                d.ctrl[mj_idx] = act_np[lab_idx]

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
