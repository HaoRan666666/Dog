# RP_wd 深度感知跨沟框架

任务名：`RP_wd_Walk_Gap_Depth`（训练）、`RP_wd_Walk_Gap_Depth_Play`（自动评估）。
这是可继续调参的训练框架，尚未验证收敛或实机跨沟能力。

## 代码入口

- `walk_env_gap_depth_cfg.py`：场景、指令、奖励、终止与训练/评估配置。
- `mdp/gap.py`：成功判定、落沟/越界检测、固定成功奖励、地形升级。
- `Dog/assets/terrain/gap_terrain.py`：两岸和沟底组成的单沟 mesh。
- `agents/rsl_rl_ppo_cfg.py::PPORunnerGapDepthCfg`：CNN + PPO，独立日志目录。
- `scripts/tools/check_gap_depth.py`：真实仿真中的张量与策略推理检查。

## 任务定义

每块地形前进方向 4 米、横向 4 米，每块有一条独立矩形沟，沟深固定 1 米。
沟的横向开口长 1.6 米，两侧各保留 1.2 米实体边缘；相邻地块的沟由 2.4 米实体隔开。
加宽地块用于减少邻近沟壑进入深度相机视野，机器人通行范围仍为中心线两侧 0.45 米。
出生原点位于地块 x=0.6 米、横向中心，前方 0.9 米为近岸边缘（地块 x=1.5 米）。
最宽沟的远岸为 x=2.5 米，保留 1.5 米落地区域。

训练地形为 20 行 × 8 列；代码等级 0–19 对应第 1–20 级，
沟宽严格使用 `0.2 + 0.8 * level / 19` 米，每级增加约 0.0421 米。
生成器的行内随机难度量化为行号，同级沟宽相同，最高级精确为 1.0 米。
默认从第 0 级开始；在当前级远岸成功落地升一级，失败或超时降一级。
最低级失败保持第 0 级，最高级成功保持第 19 级，初次 reset 不升降。
课程更新由每个环境各自完成，不按累计移动距离或全体平均成绩升级。

机器人固定出生在当前地形原点，使用默认站立高度、面向世界 +X，初始机身速度为零。
参考 platform 坑地形，只下发 0.6–1.0 m/s 前进速度，横向速度及转向角速度指令均为零，关闭 heading 控制。
这不锁死物理自由度；训练中仍可能偏移，由速度跟踪、中心线惩罚和越界终止约束。
相对出生原点前进至“0.9m + 当前级沟宽 + 0.65m”，保持直立且至少两个轮足触地时成功。
第 0 级目标为前进 1.75 米，第 19 级为 2.55 米；越界时不计成功。
成功奖励每回合 10 分，并立即终止，避免在终点重复刷奖励。
机身高度低于 0.12 米、侧向偏离达到 0.45 米、后退/前进越界或机身接触会结束回合。
前后越界线为相对出生点 x=-0.25/3.05 米，给地块边缘保留 0.35 米机身余量。
横向限制将机器人约束在沟口中部，防止通过实体侧边绕过沟。
8 秒仍未通过则超时。此处落地判定是框架基线，后续可增加四轮落地及稳定时间要求。

Actor 输入本体感觉的 3 步历史和 `(N,8,48,64)` 深度历史帧；Critic 额外使用高度扫描。
深度相机挂在 `base_link`，下俯 25°、量程 2 米、50 Hz，复用平台任务的噪声及 37 帧缓存。
Actor 不接收沟宽、地形等级或高度扫描。地形坐标仅用于训练奖励、终止和课程。
这里使用 raycast 深度相机；渲染录像时再添加 `--enable_cameras`。

## 使用

在已安装本项目的 Isaac Lab Python 环境下，从仓库根目录执行：

```bash
conda activate env_isaaclab

# 先检查仿真创建、深度形状、动作推理与奖励有限性
python scripts/tools/check_gap_depth.py --headless --num_envs 2 --steps 50

# 少量环境验证 PPO 训练链路
python scripts/rsl_rl/train.py --task RP_wd_Walk_Gap_Depth --headless --num_envs 8 --max_iterations 2

# 正式训练（按显存调整并行数量）
python scripts/rsl_rl/train.py --task RP_wd_Walk_Gap_Depth --headless --num_envs 256

# 自动前进评估：替换为实际 checkpoint 路径
python scripts/rsl_rl/play.py --task RP_wd_Walk_Gap_Depth_Play --num_envs 4 --checkpoint /path/to/model.pt
```

日志位于 `logs/rsl_rl/RP_wd_walk_gap_depth/`。Play 默认第 0 级并关闭课程，
保留物理随机化和深度延迟，关闭随机深度增强及 policy 本体噪声。
要固定评估沟宽，可在 Play 配置中设置
`self.scene.terrain.terrain_generator.sub_terrains["gap"].gap_width_range = (0.4, 0.4)`；
该设置应在 `super().__post_init__()` 前完成，以验证落地区域是否足够。
若按等级评估，可在 Play 配置中设置 `max_init_terrain_level`；此参数是随机初始等级上限，
不会将所有环境固定在该等级。固定沟宽评估优先使用上述相同上下限。

初期关注 `Episode_Termination/success`、`Episode_Termination/gap_failure`、
`Episode_Termination/base_contact` 和地形等级；终止日志是每批回合统计，
严谨评估应额外按固定沟宽统计完整回合的成功比例。
先验证相机能看到近岸沟沿与远岸，再调整速度、沟宽和奖励。
平地 MLP checkpoint 与这里的 CNN 输入结构不同，不能直接作为完整网络恢复。
本配置复用现有观测/编码器和执行器延迟实现；修改共享视觉模块会影响本任务。
