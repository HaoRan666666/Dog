# sim2sim 部署 + Foxglove 可视化

把 Isaac Lab 训练好的 RP_wd 策略部署到 MuJoCo 仿真，并通过 Foxglove 实时可视化状态。

## 目录结构

| 文件 | 作用 |
|------|------|
| `lab2mujoco2.py` | 部署入口：加载策略、跑 MuJoCo 主循环、调用下面两个模块 |
| `foxglove_bridge.py` | Foxglove 数据发布桥（可独立复用），把 MuJoCo 状态发布成 topic |
| `gamepad_simple.py` | Linux 手柄输入（读 `/dev/input/jsX`），输出 SE(2) 速度指令 |

依赖：`foxglove` Python SDK 已装在 conda 环境 **`mj`** 中（`env_isaaclab` 没有）。

---

## 运行

```bash
# 键盘模式：零指令站立（机器人原地维持平衡）
python lab2mujoco2.py --input keyboard

# 手柄模式：左摇杆移动，右摇杆左右转向
python lab2mujoco2.py --input gamepad
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | `/home/rp/model_server/rpwd_0816_platform/exported/policy.pt` | TorchScript 策略路径 |
| `--input` | `keyboard` | `keyboard`（站立）或 `gamepad`（遥控） |
| `--device` | `/dev/input/js0` | 手柄设备路径 |
| `--vx` / `--vy` / `--wz` | `1.0` / `1.0` / `1.0` | 前进 / 横移 / 转向灵敏度 |
| `--foxglove-host` | `0.0.0.0` | Foxglove WebSocket 监听地址（`0.0.0.0` = 局域网可连） |
| `--foxglove-port` | `8765` | Foxglove WebSocket 端口 |
| `--no-foxglove` | 关 | 关闭 Foxglove 数据发布 |

### 手柄按键映射

- **左摇杆前后** → 前进 / 后退
- **左摇杆左右** → 横移
- **右摇杆左右** → 转向（原地旋转）

> 手柄轴映射已按 **8BitDo Ultimate 2C** 校准：该手柄用 `ABS_Z` 报右摇杆左右（js 轴 2）、
> `ABS_RZ` 报右摇杆前后（js 轴 3），与 Xbox 360 的轴布局不同。
> 若换其他手柄出现「转向变前后」的问题，改 `gamepad_simple.py` 里 `advance()` 的轴索引即可。

---

## Foxglove 可视化

`lab2mujoco2.py` 启动时会自动在 `0.0.0.0:8765` 起一个 WebSocket 服务器，每 50Hz 发布一次状态。

### 发布的 topic

| topic | schema | 内容 |
|-------|--------|------|
| `/tf` | `foxglove.FrameTransform` | 机身 + 4 足端位姿（3D 面板用） |
| `/joint_states` | `foxglove.JointStates` | 16 关节数组（URDF/3D 用） |
| `/joints/<NAME>` ×16 | `foxglove.JointState` | 每关节一个 topic，Plot 面板按名选择 |
| `/imu/orientation` | `foxglove.Quaternion` | 姿态四元数（xyzw） |
| `/imu/angular_velocity` | `foxglove.Vector3` | 角速度 |
| `/imu/linear_acceleration` | `foxglove.Vector3` | 线加速度 |
| `/imu/projected_gravity` | `foxglove.Vector3` | 投影重力（机体系单位向量） |
| `/scene` | `foxglove.SceneUpdate` | 简易骨架（base 立方体 + 4 条腿折线） |

16 个关节名：`LF/RF/LB/RB` × `ABAD/HIP/KENN/FOOT`（MuJoCo 原生顺序）。

### 本机连接

Foxglove Studio → **Open connection** → 类型选 **Foxglove WebSocket** → URL 填：

```
ws://127.0.0.1:8765
```

### 另一台电脑连接

另一台电脑不需要装 MuJoCo / conda / Python，只需装 Foxglove Studio（桌面版）。

1. 下载桌面版：https://foxglove.dev/download
2. 确认两台机器同一局域网，且能 `ping 10.43.16.55`（A 机 IP，若变了以实际为准）
3. **Open connection** → **Foxglove WebSocket** → URL 填 `ws://10.43.16.55:8765`
4. 点 **Open**

> 网页版 https://studio.foxglove.dev 也能用，但它是 https 页面，浏览器会以「混合内容」为由
> 拦截 `ws://` 明文连接，所以局域网明文 WebSocket 建议用桌面版。

### 加面板看数据（Layout → Add panel）

| 想看什么 | 面板 | 配置 |
|---------|------|------|
| 机器人姿态 / 骨架 / 足端 | **3D** | 选 `/scene`、`/tf`、`/joint_states` |
| IMU 四元数 / 角速度 / 线加速度 / 投影重力 | **Plot** | 加 series，选 `/imu/*` 的 x/y/z/w 字段 |
| 某个关节角度 | **Plot** | 直接选 `/joints/LF_HIP_JOINT` 的 `position` |
| 任意 topic 原始报文 | **Raw Messages** | 选对应 topic |

调好布局后 **Save layout**，下次打开自动复用。

---

## 常见问题

- **另一台电脑连不上**：确认 A 机服务器在跑、`--foxglove-host` 是 `0.0.0.0`；B 机 `ping 10.43.16.55`。
- **连上了没数据**：A 机 MuJoCo 主循环在跑才有数据（`publish()` 每 50Hz 调一次），暂停/关掉仿真就没数据了。
- **端口被占用**：启动时会打印 `[Foxglove] 启动失败 (端口被占用?)`，换 `--foxglove-port` 或在 B 机填对应端口。
- **IP 变了**：A 机 DHCP 分配 IP 重启后可能变化，B 机 URL 要跟着改，或给 A 机设静态 IP。
- **手柄转向变前后**：轴映射问题，见上文「手柄按键映射」。
