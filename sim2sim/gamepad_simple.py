"""Linux 游戏手柄输入设备：直接读取 /dev/input/jsX，输出 SE(2) 速度指令。

独立于 sim2sim 部署脚本，供 lab2mujoco2.py 等入口按需调用。
"""

import os
import struct


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
