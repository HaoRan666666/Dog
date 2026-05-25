"""Gamepad / Joystick 控制器 —— 读取手柄输入，输出速度指令 (v_x, v_y, omega_z).

用法（在 Isaac Sim play 脚本中）::

    from my_utils.joy import GamepadController

    joy = GamepadController()
    while sim.is_running():
        command = joy.advance()          # → torch.Tensor([v_x, v_y, omega_z])
        env.command_manager.get_term("base_velocity").vel_command_b[:] = command
        env.step(actions)
    joy.close()
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
import weakref
from dataclasses import dataclass


def get_gamepad_controller(
    sensitivity_linear: float = 1.0,
    sensitivity_angular: float = 1.0,
    dead_zone: float = 0.05,
    device: str = "cpu",
) -> "GamepadController | KeyboardFallback":
    """工厂函数：优先返回 GamepadController，没手柄时返回 KeyboardFallback。

    Args:
        sensitivity_linear:   线速度灵敏度 (v_x, v_y) 倍率。
        sensitivity_angular:  角速度灵敏度 (omega_z) 倍率。
        dead_zone:            摇杆死区阈值。
        device:               torch device 字符串。

    Returns:
        GamepadController 或 KeyboardFallback。
    """
    try:
        return GamepadController(
            sensitivity_linear=sensitivity_linear,
            sensitivity_angular=sensitivity_angular,
            dead_zone=dead_zone,
            device=device,
        )
    except RuntimeError:
        print("[joy] 未检测到手柄，回退到键盘控制 (WASD + QE)。")
        return KeyboardFallback(
            sensitivity_linear=sensitivity_linear,
            sensitivity_angular=sensitivity_angular,
            device=device,
        )


# ── 手柄控制器 ──────────────────────────────────────────────────────


class GamepadController:
    """读取 Carbon/Omniverse 手柄事件，生成 SE(2) 速度指令。

    摇杆映射:
        - 左摇杆 上/下    → v_x  (+/-)
        - 左摇杆 右/左    → v_y  (+/-)
        - 右摇杆 右/左    → omega_z  (+/-)
    """

    def __init__(
        self,
        sensitivity_linear: float = 1.0,
        sensitivity_angular: float = 1.0,
        dead_zone: float = 0.05,
        device: str = "cpu",
    ):
        import carb
        import omni

        self._sensitivity_linear = sensitivity_linear
        self._sensitivity_angular = sensitivity_angular
        self._dead_zone = dead_zone
        self._device = device

        # 关闭 Isaac Sim 默认的游戏手柄相机控制
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)

        # 获取手柄接口
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)

        if self._gamepad is None:
            raise RuntimeError("No gamepad detected.")

        print(f"[joy] 检测到手柄: {self._input.get_gamepad_name(self._gamepad)}")

        # 订阅手柄事件
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

        # 原始缓冲: (positive, negative) × (v_x, v_y, omega_z)
        self._raw = np.zeros((2, 3), dtype=np.float32)
        # 额外回调
        self._callbacks: dict[int, Callable[[], None]] = {}

        self._setup_mappings()

    # ── 公开 API ─────────────────────────────────────────────────

    def advance(self) -> torch.Tensor:
        """返回当前速度指令 tensor，形状 (3,) → (v_x, v_y, omega_z)。"""
        # 正负通道取最大值，正通道被负通道压制 → 负号
        sign = self._raw[1, :] > self._raw[0, :]  # True → 负
        vals = self._raw.max(axis=0)
        vals[sign] *= -1
        return torch.from_numpy(vals).float().to(self._device)

    def reset(self) -> None:
        """清零所有指令。"""
        self._raw.fill(0.0)

    def add_callback(self, button: str, func: Callable[[], None]) -> None:
        """注册手柄按键回调。

        Args:
            button: 按键名，如 "A", "B", "X", "Y", "LEFT_BUMPER", "RIGHT_BUMPER", "START", "BACK".
            func:   按下时调用的无参函数。
        """
        try:
            btn_id = getattr(carb.input.GamepadInput, button.upper())
            self._callbacks[btn_id] = func
        except AttributeError:
            print(f"[joy] 未知按键: {button}")

    def close(self) -> None:
        """取消订阅，释放资源。"""
        if self._gamepad_sub is not None:
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None

    def __del__(self) -> None:
        self.close()

    def __str__(self) -> str:
        name = self._input.get_gamepad_name(self._gamepad) if self._gamepad else "none"
        return (
            f"GamepadController(device={name}, "
            f"lin={self._sensitivity_linear}, ang={self._sensitivity_angular}, "
            f"dead_zone={self._dead_zone})"
        )

    # ── 内部 ────────────────────────────────────────────────────

    def _setup_mappings(self) -> None:
        import carb

        # 手柄摇杆方向 → (通道, 轴, 灵敏度)
        self._STICK_MAP: dict[int, tuple[int, int, float]] = {
            carb.input.GamepadInput.LEFT_STICK_UP:    (0, 0,  self._sensitivity_linear),
            carb.input.GamepadInput.LEFT_STICK_DOWN:  (1, 0,  self._sensitivity_linear),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1,  self._sensitivity_linear),
            carb.input.GamepadInput.LEFT_STICK_LEFT:  (1, 1,  self._sensitivity_linear),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT:(0, 2,  self._sensitivity_angular),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 2,  self._sensitivity_angular),
        }

    def _on_event(self, event, *args, **kwargs) -> bool:
        val = event.value
        if abs(val) < self._dead_zone:
            val = 0.0

        if event.input in self._STICK_MAP:
            direction, axis, sensitivity = self._STICK_MAP[event.input]
            self._raw[direction, axis] = sensitivity * val

        if event.input in self._callbacks:
            self._callbacks[event.input]()

        return True


# ── 键盘回退（无手柄时使用） ──────────────────────────────────────


class KeyboardFallback:
    """当没有手柄时，使用键盘模拟 SE(2) 速度指令。

    按键:
        W / S      →  v_x  (+/-)
        D / A      →  v_y  (+/-)
        Q / E      →  omega_z  (+/-)
        Space      →  重置所有指令
    """

    def __init__(
        self,
        sensitivity_linear: float = 1.0,
        sensitivity_angular: float = 1.0,
        device: str = "cpu",
    ):
        import carb
        import omni

        self._sensitivity_linear = sensitivity_linear
        self._sensitivity_angular = sensitivity_angular
        self._device = device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

        self._command = np.zeros(3, dtype=np.float32)
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._setup_mappings()

    def advance(self) -> torch.Tensor:
        return torch.from_numpy(self._command.copy()).float().to(self._device)

    def reset(self) -> None:
        self._command.fill(0.0)

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._callbacks[key.upper()] = func

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __del__(self) -> None:
        self.close()

    def __str__(self) -> str:
        return f"KeyboardFallback(lin={self._sensitivity_linear}, ang={self._sensitivity_angular})"

    def _setup_mappings(self) -> None:
        self._KEY_MAP: dict[str, np.ndarray] = {
            "W":     np.array([ 1.0,  0.0,  0.0]) * self._sensitivity_linear,    # 前进
            "S":     np.array([-1.0,  0.0,  0.0]) * self._sensitivity_linear,    # 后退
            "D":     np.array([ 0.0,  1.0,  0.0]) * self._sensitivity_linear,    # 右移
            "A":     np.array([ 0.0, -1.0,  0.0]) * self._sensitivity_linear,    # 左移
            "E":     np.array([ 0.0,  0.0,  1.0]) * self._sensitivity_angular,   # 右转
            "Q":     np.array([ 0.0,  0.0, -1.0]) * self._sensitivity_angular,   # 左转
        }

    def _on_event(self, event, *args, **kwargs) -> bool:
        import carb
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name.upper()
            if name == "SPACE":
                self.reset()
            elif name in self._KEY_MAP:
                self._command += self._KEY_MAP[name]
            elif name in self._callbacks:
                self._callbacks[name]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            name = event.input.name.upper()
            if name in self._KEY_MAP:
                self._command -= self._KEY_MAP[name]
        return True
