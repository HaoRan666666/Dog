# Dog

基于 **Isaac Lab / Isaac Sim** 的机器人运动控制强化学习项目，使用 **RSL-RL PPO** 训练策略，并提供 **MuJoCo sim2sim** 验证工具。

项目包含 RP_wd 轮腿机器人、RPMini 双足机器人，以及 L1、BPX、RCdog 四足机器人的任务配置。其中 RP_wd 支持平地、台阶、平台、双轮足站立行走和深度感知任务。

## 主要功能

- **并行强化学习训练**：通过 Isaac Lab 的 Manager-Based 环境组织场景、观测、动作、奖励、随机化和课程学习。
- **四轮足运动控制**：RP_wd 包含 12 个腿部关节和 4 个轮关节，配置了执行器延迟。
- **地形课程**：支持台阶、平台和跨沟地形，相关生成配置位于 `assets/terrain/`。
- **深度感知策略**：使用带噪声和延迟的 raycast 深度相机，通过 CNN 编码深度历史帧，结合本体观测输出动作。
- **策略回放与导出**：支持键盘、手柄控制，以及普通策略的 TorchScript / ONNX 导出和深度策略的分模块 ONNX 导出。
- **跨仿真验证**：在 MuJoCo 中运行 RP_wd 策略，并可通过 Foxglove 查看机器人状态。

> 跨沟任务目前是可继续调参的训练框架，尚未验证收敛或实机跨沟能力。详见[跨沟任务说明](source/Dog/Dog/tasks/manager_based/RP_wd/README_gap_depth.md)。

## 目录结构

```text
Dog/
├── source/Dog/                       # 可编辑安装的 Python 扩展
│   ├── setup.py
│   ├── config/extension.toml
│   └── Dog/
│       ├── assets/                  # USD、URDF、MJCF 机器人资源及地形
│       ├── robots/                  # 机器人初始状态、关节与执行器配置
│       ├── tasks/manager_based/     # RP_wd、RPMini、L1、bpx、rcdog 任务
│       ├── policies/               # 深度编码器策略及 RSL-RL 兼容补丁
│       ├── sensor/                 # 分组射线投射与带噪声相机
│       └── utils/                  # 延迟缓存、噪声、数学及 Warp 工具
├── scripts/
│   ├── rsl_rl/                     # 训练、回放和命令行参数
│   ├── keyboard_play.py            # Isaac Sim 键盘 / 手柄遥控
│   ├── zero_agent.py               # 零动作环境检查
│   ├── random_agent.py             # 随机动作环境检查
│   └── tools/                      # 跨沟检查、TensorBoard 数据导出
├── sim2sim/                        # MuJoCo 运行、ONNX 导出及 Foxglove 桥接
└── my_utils/                       # 手柄等辅助工具
```

## 环境准备

先准备能运行 Isaac Lab 的 Python 环境，再安装本项目。`setup.py` 只声明了少量 Python 依赖，**不会自动安装 Isaac Sim、Isaac Lab 或完整训练依赖**。

| 组件 | 当前代码要求或用途 |
| --- | --- |
| Python | 包元数据要求 `>=3.10`；具体版本需与所用 Isaac Lab / Isaac Sim 匹配 |
| Isaac Sim、Isaac Lab | 仿真、环境管理、传感器及机器人资源加载 |
| `isaaclab_rl`、`isaaclab_tasks` | RSL-RL 接口、任务注册及配置工具 |
| PyTorch、RSL-RL | 策略训练；训练入口检查 `rsl-rl-lib >= 3.0.1` |
| TensorBoard | 查看训练指标及导出标量数据 |
| ONNX 相关依赖 | 策略导出；独立数值对比脚本还需要 `onnxruntime` |
| MuJoCo | 可选，用于 `sim2sim/` |

仓库尚未提供完整的依赖锁定文件或经验证的版本组合；包元数据中的版本标签不代表全部任务均已兼容验证。优先使用已经能运行本项目所需 Isaac Lab 接口的环境。

以下命令均在**仓库根目录**执行，并使用该环境中的 Python：

```bash
python -m pip install -e source/Dog
```

如果使用 Isaac Lab 自带的启动脚本，将后续命令中的 `python` 替换为对应解释器入口，例如：

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/Dog
```

## 支持的任务

任务 ID 区分大小写，以各任务目录下的 `__init__.py` 注册内容为准。

### RP_wd

下表每个训练任务均有对应的回放任务：在训练 ID 后添加 `_Play`，例如 `RP_wd_Walk_Terrain_Play`。

| 训练任务 ID | 内容 | 日志实验目录名 |
| --- | --- | --- |
| `RP_wd_Walk_Flat` | 平地轮腿行走 | `RP_wd_walk_flat` |
| `RP_wd_Walk_Terrain` | 台阶地形；Actor 使用本体观测，Critic 额外使用高度扫描 | `RP_wd_walk_terrain` |
| `RP_wd_Walk_Platform` | 平台地形 | `RP_wd_walk_platform` |
| `RP_wd_Walk_Biped` | 仅用两条后腿站立的双轮足行走 | `RP_wd_walk_biped` |
| `RP_wd_Walk_Stair` | 深度感知台阶任务 | `RP_wd_walk_stair` |
| `RP_wd_Walk_Platform_Depth` | 深度感知平台任务 | `RP_wd_walk_platform_depth` |
| `RP_wd_Walk_Gap_Depth` | 深度感知跨沟任务 | `RP_wd_walk_gap_depth` |

### 其他机器人

| 机器人 | 训练任务 ID | 平地回放任务 ID | 额外地形回放任务 ID |
| --- | --- | --- | --- |
| RPMini | `RPMini_Walk_Flat` | `RPMini_Walk_Flat_Play` | — |
| L1 | `L1_Walk_Flat-v0` | `L1_Walk_Flat-v0_Play` | `L1_Terrain_Play-v0` |
| BPX | `BPX_Walk_Flat-v0` | `BPX_Walk_Flat-v0_Play` | `BPX_Terrain_Play-v0` |
| RCdog | `RCdog_Walk_Flat-v0` | `RCdog_Walk_Flat-v0_Play` | `RCdog_Terrain_Play-v0` |

`scripts/list_envs.py` 目前仍筛选名称包含 `Template-` 的任务，因此不能用于完整列出本项目任务；请参考上述表格或注册文件。

## 快速开始

### 1. 检查环境

先用少量环境检查资源加载和仿真是否正常：

```bash
python scripts/zero_agent.py --task RP_wd_Walk_Flat --num_envs 4

# 也可以使用随机动作检查环境
python scripts/random_agent.py --task RP_wd_Walk_Flat --num_envs 4
```

零动作和随机动作脚本用于检查环境，不代表训练后的运动效果。没有图形界面时可添加 `--headless`。

### 2. 训练

先进行少量迭代，检查训练链路：

```bash
python scripts/rsl_rl/train.py \
  --task RP_wd_Walk_Terrain \
  --num_envs 8 \
  --max_iterations 2 \
  --headless
```

地形任务训练示例，环境数量和迭代次数可按实验需求调整：

```bash
python scripts/rsl_rl/train.py \
  --task RP_wd_Walk_Terrain \
  --num_envs 1024 \
  --max_iterations 10000 \
  --run_name terrain \
  --logger tensorboard \
  --headless
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--task` | 选择注册的任务 |
| `--num_envs` | 覆盖并行环境数量 |
| `--max_iterations` | 覆盖训练迭代次数 |
| `--seed` | 设置随机种子 |
| `--headless` | 关闭图形界面 |
| `--run_name` | 为本次运行目录添加后缀 |
| `--logger tensorboard` | 使用 TensorBoard 记录指标 |
| `--video` | 录制视频；训练和回放脚本会自动启用渲染相机 |

RP_wd 基础 PPO 配置当前的 `max_iterations` 为 `150`，多数派生任务沿用该值；跨沟任务单独配置为 `5000`。正式实验请明确设置所需迭代次数，示例数值不保证策略收敛。

### 3. 查看日志和恢复训练

训练结果保存在：

```text
logs/rsl_rl/<experiment_name>/<时间戳>[_<run_name>]/
├── model_*.pt          # 训练 checkpoint
├── params/
│   ├── env.yaml        # 本次环境配置
│   └── agent.yaml      # 本次算法配置
└── videos/train/       # 启用训练录像时生成
```

查看指标或导出 CSV：

```bash
tensorboard --logdir logs/rsl_rl
python scripts/tools/export_tb.py logs/rsl_rl/RP_wd_walk_terrain/
```

恢复训练时，将运行目录名和模型文件名替换为实际值：

```bash
python scripts/rsl_rl/train.py \
  --task RP_wd_Walk_Terrain \
  --resume \
  --load_run '<运行目录名>' \
  --checkpoint 'model_1000.pt' \
  --num_envs 1024 \
  --max_iterations 10000 \
  --headless
```

训练入口通过实验目录、`--load_run` 和 `--checkpoint` 查找模型；回放入口的 `--checkpoint` 则可直接指定模型路径。恢复时需保持网络结构和观测配置兼容。

### 4. 回放和遥控

将 `/path/to/model.pt` 替换为训练得到的 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task RP_wd_Walk_Terrain_Play \
  --num_envs 1 \
  --checkpoint /path/to/model.pt \
  --real-time
```

使用键盘控制速度指令：

```bash
python scripts/keyboard_play.py \
  --task RP_wd_Walk_Terrain_Play \
  --checkpoint /path/to/model.pt
```

| 按键 | 功能 |
| --- | --- |
| `↑` / `↓` | 前进 / 后退 |
| `←` / `→` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 停止 |
| `Enter` | 复位 |

添加 `--input gamepad` 可改用手柄。键盘控制需要图形窗口，不要添加 `--headless`。`play.py` 运行环境生成的指令，交互遥控请使用 `keyboard_play.py`。

RP_wd 地形和平台的 Play 任务目前仍绑定平地 Runner 配置，自动查找模型时会使用平地日志目录；遥控脚本的默认查找目录也固定为平地目录。因此回放示例均显式传入 `--checkpoint`。

### 5. 策略导出

`scripts/rsl_rl/play.py` 加载模型后会自动导出到 checkpoint 同级的 `exported/` 目录：

| 策略类型 | 当前导出文件 |
| --- | --- |
| 普通 MLP 策略 | `policy.pt`（TorchScript）、`policy.onnx` |
| 深度编码器策略 | 默认 `0-depth_image.onnx`（编码器）、`actor.onnx` |

深度策略需要先编码深度图，再按训练时的顺序拼接本体观测和深度特征。普通训练 checkpoint、TorchScript 文件和 ONNX 文件用途不同，不能互换。

## 深度感知任务

台阶、深度平台和跨沟任务使用项目内的 `EncoderActorCritic`，相关配置位于：

- [深度台阶环境](source/Dog/Dog/tasks/manager_based/RP_wd/walk_env_stair_cfg.py)
- [深度平台环境](source/Dog/Dog/tasks/manager_based/RP_wd/walk_env_preceptive_platform_cfg.py)
- [跨沟环境](source/Dog/Dog/tasks/manager_based/RP_wd/walk_env_gap_depth_cfg.py)
- [策略与 PPO 配置](source/Dog/Dog/tasks/manager_based/RP_wd/agents/rsl_rl_ppo_cfg.py)

跨沟任务可先运行专用检查，再开始训练：

```bash
python scripts/tools/check_gap_depth.py --headless --num_envs 2 --steps 50

python scripts/rsl_rl/train.py \
  --task RP_wd_Walk_Gap_Depth \
  --num_envs 256 \
  --max_iterations 5000 \
  --headless
```

这些任务使用 raycast 深度相机，深度观测本身不要求 `--enable_cameras`；渲染录像时由 `--video` 自动启用。深度图历史帧会增加显存开销，可从较少并行环境开始。

训练和回放入口已调用 `patch_rollout_storage()`，用于兼容分项字典观测。新增入口时需保留这一步。平地 MLP 与深度 CNN 策略结构不同，不能直接用同一个完整 checkpoint 恢复训练。

## MuJoCo sim2sim

`sim2sim/` 提供 RP_wd 的 MuJoCo 运行脚本，机器人模型位于 `source/Dog/Dog/assets/RP_wd/mjcf/RP_wd.xml`。可在单独的 Python 环境中准备 PyTorch、NumPy 和 MuJoCo；Foxglove 可视化及 ONNX 工具另需各自依赖。

普通策略使用回放脚本导出的 **TorchScript `policy.pt`**：

```bash
python sim2sim/lab2mujoco2.py \
  --ckpt /path/to/exported/policy.pt \
  --input gamepad \
  --no-foxglove
```

该脚本还支持通过 `--biped-ckpt`、`--platform-ckpt`、`--step-ckpt` 加载其他策略。默认模型路径包含开发者机器的绝对路径，使用前应替换为本机文件；默认平地模型必须成功加载。

深度台阶策略使用原始训练 checkpoint：

```bash
python sim2sim/lab2mujoco_stair.py \
  --ckpt /path/to/stair/model.pt \
  --input gamepad \
  --no-foxglove
```

MuJoCo 脚本中的 `--input keyboard` 当前表示零速度指令站立，交互移动使用 Linux 手柄。具体输入映射及 Foxglove 连接方式见 [sim2sim 说明](sim2sim/README.md)；其中默认路径和参数请以当前脚本为准。

另有无需启动 Isaac Sim 的台阶策略 ONNX 导出与数值对比工具：

```bash
python sim2sim/export_stair_onnx.py \
  --ckpt /path/to/stair/model.pt \
  --out /path/to/exported_onnx

python sim2sim/run_stair_onnx.py \
  --ckpt /path/to/stair/model.pt \
  --onnx_dir /path/to/exported_onnx
```

这组工具依照 `sim2sim/encoder_policy.py` 中固定的台阶网络和观测维度加载模型，输出 `depth_encoder.onnx` 与 `actor.onnx`。文件名和输入接口与 `play.py` 的导出不同，应配套使用；修改网络或观测后，需要同步更新这些工具。

## 配置修改入口

| 修改目标 | 文件或目录 |
| --- | --- |
| 机器人模型、初始姿态、执行器参数 | `source/Dog/Dog/robots/` |
| RP_wd 平地场景、动作及基础观测 | `source/Dog/Dog/tasks/manager_based/RP_wd/walk_env_cfg.py` |
| RP_wd 地形场景、指令和课程 | `source/Dog/Dog/tasks/manager_based/RP_wd/walk_env_terrain_cfg.py` |
| 台阶尺寸、类型比例及难度范围 | `source/Dog/Dog/assets/terrain/step_terrain.py` |
| 奖励、终止、随机化与课程实现 | 各任务目录下的 `mdp/` |
| PPO 超参数和实验目录名 | 各任务目录下的 `agents/rsl_rl_ppo_cfg.py` |
| 新任务注册 | 各任务目录下的 `__init__.py` |

当前 RP_wd 地形训练配置默认使用 2048 个并行环境，仿真步长为 `0.0025 s`，`decimation=8`，对应 **50 Hz 控制频率**。Actor 继承平地本体观测，Critic 额外获得高度扫描。

`RP_wd_Walk_Terrain_Play` 当前仅保留倒金字塔下台阶地形，关闭课程和机身接触终止，初始地形等级上限为 15。因此 Play 场景与训练分布不同，评估其他地形时应调整对应配置。

## 常见问题

- **找不到 `isaaclab` 或 `Dog`**：确认使用的是 Isaac Lab 对应的 Python，并在该环境中执行过 `python -m pip install -e source/Dog`。
- **任务列表为空**：检查 `scripts/list_envs.py` 的 `Template-` 筛选条件；任务是否存在以注册文件为准。
- **回放找不到模型**：显式指定 `--checkpoint`，确认任务、模型结构及观测维度匹配。
- **显存不足**：降低 `--num_envs`，深度任务尤其需要控制并行规模；检查是否启用了录像或相机调试可视化。
- **修改实验目录名未生效**：当前 `--experiment_name` 虽已声明，但参数更新函数未应用它；可修改 Runner 配置或在训练命令末尾添加 Hydra 覆写 `agent.experiment_name=my_experiment`。
- **MuJoCo 找不到策略文件**：替换脚本中开发者机器的默认路径，并区分 `--ckpt` 接收的是 TorchScript 还是原始 checkpoint。
