#!/usr/bin/env python3
"""Isaac Sim 策略播放脚本：支持键盘或手柄控制四足机器人。

用法:
    # 键盘控制
    python sim2sim/keyboard_play.py --task BPX_Walk_Flat-v0_Play \
        --checkpoint logs/rsl_rl/bpx_walk_flat/xxx/model_9999.pt

    # 手柄控制
    python sim2sim/keyboard_play.py --task BPX_Walk_Flat-v0_Play \
        --checkpoint logs/rsl_rl/bpx_walk_flat/xxx/model_9999.pt --input gamepad

键盘按键:
    ↑ 前进    ↓ 后退    ← 左移    → 右移
    Q 左转    E 右转    Space 停止    Enter 复位

手柄映射:
    左摇杆 上/下 → 前进/后退
    左摇杆 左/右 → 左移/右移
    右摇杆 左/右 → 左转/右转
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from isaaclab.app import AppLauncher

# ── 命令行参数 ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True,
                    help="Gym 环境 ID，如 BPX_Walk_Flat-v0_Play")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="策略 checkpoint 路径")
parser.add_argument("--input", type=str, default="keyboard",
                    choices=["keyboard", "gamepad"],
                    help="输入设备: keyboard 或 gamepad")
parser.add_argument("--vx", type=float, default=1.0,
                    help="前进灵敏度")
parser.add_argument("--vy", type=float, default=1.0,
                    help="横移灵敏度")
parser.add_argument("--wz", type=float, default=2.0,
                    help="转向灵敏度")
parser.add_argument("--terrain-level", type=int, default=None,
                    help="固定地形等级 (0~29)，不指定则随机")
parser.add_argument("--record-torque", type=str, default=None,
                    help="记录关节扭矩，保存为 NPY 并出图 (指定输出文件名前缀，如 torque_log)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── 启动 Isaac Sim ─────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ═══════════════════════════════════════════════════════════════════
# 以下代码运行在 Isaac Sim 进程中，carb / omni / pxr 此时可用
# ═══════════════════════════════════════════════════════════════════

import time
import weakref
import numpy as np
import torch
import gymnasium as gym

import carb   # Omniverse 底层接口
import omni   # Omniverse 应用层接口

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import isaaclab_tasks  # noqa: F401  注册官方任务
import Dog.tasks        # noqa: F401  注册自定义任务

import matplotlib
matplotlib.use("Agg")  # 无头环境，不依赖显示器
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# 四元数工具
# ═══════════════════════════════════════════════════════════════════

def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q=[w,x,y,z] 旋转向量 v，返回旋转后的向量。"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


# ═══════════════════════════════════════════════════════════════════
# 键盘输入设备
# ═══════════════════════════════════════════════════════════════════

class Se2KeyboardSimple:
    """通过 Omniverse 键盘接口读取按键，输出 SE(2) 速度指令 (v_x, v_y, ω_z)。

    按键按下时持续输出对应方向的速度，松开后归零。
    """

    def __init__(self, vx=1.0, vy=1.0, wz=2.0):
        # 获取 Omniverse 应用窗口和输入接口
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # 订阅键盘事件（用 weakref 避免循环引用）
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )
        self._cmd = np.zeros(3, dtype=np.float32)  # 当前速度指令
        self._keys = {}                             # 记录哪些键正被按住
        self.reset_pressed = False                  # 复位请求标志

        # 按键 → 速度增量的映射
        self._key_sensitivity = {
            "UP":    (vx,  0.0, 0.0),   # 上箭头: 前进
            "DOWN":  (-vx, 0.0, 0.0),   # 下箭头: 后退
            "LEFT":  (0.0, vy,  0.0),   # 左箭头: 左移
            "RIGHT": (0.0, -vy, 0.0),   # 右箭头: 右移
            "Q":     (0.0, 0.0, wz),    # Q: 左转
            "E":     (0.0, 0.0, -wz),   # E: 右转
        }

    def _on_event(self, event, *args):
        """键盘事件回调：按下时记录，松开时清除。"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_sensitivity:
                self._keys[event.input.name] = True
            elif event.input.name == "SPACE":
                # 空格: 紧急停止
                self._cmd.fill(0.0)
                self._keys.clear()
            elif event.input.name in ("ENTER", "RETURN", "KEY_RETURN"):
                # 回车: 复位机器人
                self.reset_pressed = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys.pop(event.input.name, None)
        return True

    def advance(self) -> np.ndarray:
        """每帧调用，返回当前累计的速度指令。"""
        self._cmd.fill(0.0)
        for name, sens in self._key_sensitivity.items():
            if self._keys.get(name):
                self._cmd += np.asarray(sens)
        return self._cmd


# ═══════════════════════════════════════════════════════════════════
# 手柄输入设备
# ═══════════════════════════════════════════════════════════════════

class GamepadSimple:
    """直接读取 Linux /dev/input/jsX 设备文件，输出 SE(2) 速度指令。

    绕过了 Omniverse 的手柄接口（有时检测不到），直接从内核驱动层读取。
    """

    def __init__(self, vx=1.0, vy=1.0, wz=2.0, dead_zone=0.1, dev="/dev/input/js0"):
        import os
        self._vx = vx
        self._vy = vy
        self._wz = wz
        self._dead_zone = dead_zone  # 死区，避免摇杆漂移
        # 非阻塞模式打开，不阻塞仿真主循环
        self._fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        self._axes = [0.0] * 8       # 轴状态（最多 8 个轴）
        self._buttons = [0] * 16     # 按键状态
        print(f"手柄已连接: {dev}")

    def _poll(self):
        """非阻塞读取手柄事件，更新轴和按键状态。"""
        import struct, os
        try:
            while True:
                data = os.read(self._fd, 8)       # 每个事件 8 字节
                if not data:
                    break
                # Linux joystick 事件格式: time(I) | value(h) | type(B) | number(B)
                _time, value, ev_type, number = struct.unpack('IhBB', data)
                ev_type &= 0x7f                     # 去掉最高位的 init 标记
                if ev_type == 2:                    # 轴事件
                    self._axes[number] = value / 32767.0   # 归一化到 [-1, 1]
                elif ev_type == 1:                  # 按键事件
                    self._buttons[number] = value
        except BlockingIOError:
            pass  # 无新事件，跳过

    def advance(self) -> np.ndarray:
        """每帧调用，返回当前摇杆对应的速度指令。"""
        self._poll()
        cmd = np.zeros(3, dtype=np.float32)
        ly = self._axes[1]    # 左摇杆 Y 轴 (向上为负)
        lx = self._axes[0]    # 左摇杆 X 轴
        rx = self._axes[3]    # 右摇杆 X 轴（用于转向）
        if abs(ly) > self._dead_zone:
            cmd[0] = -ly * self._vx    # 前进/后退（取反修正方向）
        if abs(lx) > self._dead_zone:
            cmd[1] = -lx * self._vy    # 左移/右移
        if abs(rx) > self._dead_zone:
            cmd[2] = -rx * self._wz    # 左转/右转
        return cmd


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── 环境配置 ──────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.enable_corruption = False  # Play 模式关噪声

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path("logs/rsl_rl/RP_wd_walk_flat", ".*", "model_.*.pt")
    print(f"Checkpoint: {resume_path}")

    # ── 创建环境 ──────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 强制地形等级（覆盖随机初始化）
    if args_cli.terrain_level is not None:
        terrain = env.unwrapped.scene.terrain
        level = args_cli.terrain_level
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels[:].long(),
            terrain.terrain_types[:].long(),
        ]
        print(f"地形等级设为: {level}")

    env = RslRlVecEnvWrapper(env)        # 包装为 RSL-RL 兼容格式

    # ── 加载训练好的策略 ───────────────────────────────────────────
    from isaaclab_tasks.utils.hydra import load_cfg_from_registry
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 输入设备选择 ──────────────────────────────────────────────
    if args_cli.input == "gamepad":
        input_device = GamepadSimple(vx=args_cli.vx, vy=args_cli.vy, wz=args_cli.wz)
        print("手柄控制: 左摇杆移动, 右摇杆左右转向")
    else:
        input_device = Se2KeyboardSimple(vx=args_cli.vx, vy=args_cli.vy, wz=args_cli.wz)
        print("键盘控制: ↑↓←→ 移动, Q/E 转向, Space 停止, Enter 复位")

    # ── 关闭 heading 模式 ─────────────────────────────────────────
    # 训练时 heading_command 会用航向误差覆盖 ang_vel_z，导致直接写入无效
    term = env.unwrapped.command_manager._terms["base_velocity"]
    term.cfg.heading_command = False

    # ── 扭矩记录 ──────────────────────────────────────────────────
    joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
    torque_log = [] if args_cli.record_torque else None
    if torque_log is not None:
        print(f"扭矩记录已开启，输出: {args_cli.record_torque}.npy / .png")

    # ── 主仿真循环 ───────────────────────────────────────────────
    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    step = 0

    while simulation_app.is_running():
        start = time.time()
        step += 1

        # 1) 读取手柄/键盘指令 → 写入环境速度命令
        cmd = input_device.advance()
        if isinstance(cmd, torch.Tensor):
            cmd = cmd.cpu().numpy()
        term.vel_command_b[:] = torch.tensor(cmd, dtype=torch.float32, device=env.unwrapped.device)

        # 2) 策略推理 + 环境步进
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # 2.5) 扭矩记录
        if torque_log is not None:
            t = env.unwrapped.scene["robot"].data.applied_torque[0].cpu().numpy()
            c = env.unwrapped.scene["robot"].data.computed_torque[0].cpu().numpy()
            torque_log.append(t)
            if step % 50 == 0:
                for i, name in enumerate(joint_names):
                    diff = abs(c[i] - t[i])
                    if diff > 1.0:
                        print(f"[clip] {name}: computed={c[i]:.1f} → applied={t[i]:.1f}  diff={diff:.1f}")

        # 3) 翻倒自动复位
        base_height = env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()
        if base_height < 0.15:
            with torch.inference_mode():
                env.unwrapped.reset()
            obs = env.get_observations()
            term.vel_command_b[:] = 0.0
            print("Robot flipped, auto-reset.")

        # 4) 控制频率对齐
        elapsed = time.time() - start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # ── 保存扭矩数据 + 出图 ────────────────────────────────────────
    if torque_log is not None and len(torque_log) > 0:
        data = np.stack(torque_log, axis=0)       # [N_steps, N_joints]
        t = np.arange(len(data)) * dt

        out_dir = os.path.join(_SCRIPT_DIR, f"torque_{args_cli.record_torque}")
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "data.npy"), data)
        print(f"扭矩数据已保存: {out_dir}/data.npy  (shape={data.shape})")

        for idx, name in enumerate(joint_names):
            short = name.replace("_joint", "")
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(t, data[:, idx], linewidth=0.8)
            ax.set_ylabel("Torque (Nm)")
            ax.set_xlabel("Time (s)")
            ax.set_title(short)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{short}.png"), dpi=150)
            plt.close(fig)

        print(f"扭矩曲线已保存: {out_dir}/  ({len(joint_names)} 张图)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
