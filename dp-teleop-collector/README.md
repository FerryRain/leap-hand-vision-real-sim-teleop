# LEAP 手指抓糖示教数据

这个目录的主流程只训练 LEAP Hand 的 16 个手指关节完成“抓一把糖”。相机仍实时识别人手并产生 LEAP 关节目标，但训练数据里**没有 RGB、深度图、FR3 位姿或 FR3 动作**。机械臂抬升由外部程序或人工操作负责，不属于本策略。

当前版本只采集 `grasp`：手指在糖果阻力下暂时闭不拢，随后机械臂抬起、阻力变小，电机继续追踪闭合目标并逐渐收紧。不要把释放动作混入同一数据集。

## 1. 这是不是力控

现有 LEAP 驱动不是一个根据外部力传感器闭环调节目标的“外层力控器”。它使用 Dynamixel **Mode 5（current-based position control，基于电流的位置控制）**：

1. 遥操程序发送 16 维关节位置目标；
2. 电机内部位置环持续追踪该目标；
3. `Goal Current` 限制位置环可使用的电流/力矩。

因此，糖果阻挡手指时会出现 `actual_position != goal_position`；负载减小后，即使目标没有变化，电机内部位置环仍会继续追踪原来的闭合目标，所以手指会逐渐闭拢。这种现象看起来有柔顺性，但不是本程序实现了显式力控。

采集器读取每个电机的有符号 `Present Current` 原始寄存器值。它可以反映负载变化，但**没有经过力或力矩标定**，不能直接当作牛顿或牛顿米使用。训练和执行时应保持相同的电机型号、控制模式、增益与 `Goal Current` 配置；不要为了抓得更紧而直接提高电流上限。

## 2. 策略真正看到和输出的内容

每一帧的低维观察是 48 维：

```text
robot_state = [
  actual_position_rad[16],
  finite_difference_velocity_rad_s[16],
  signed_present_current_raw[16]
]
```

策略动作是 16 维：

```text
action = post_slew_goal_position_rad[16]
```

这里的 `action` 是经过安全限位和单步变化限制后，**实际发送给电机的位置目标**，不是下一帧实测关节位置。数据还单独保存 `goal_position` 和 `position_error = goal_position - actual_position`，用于检查受阻和继续闭合的过程；它们是诊断数组，不放进默认策略观察。

底层控制约为 30 Hz，训练数据固定为 10 Hz。采集器用相邻两次真实 LEAP
反馈把本体状态插值到严格的 0.1 秒网格；动作采用当时电机上真实生效的最近一次
post-slew 目标（零阶保持），不会插值出一个从未发送过的动作。相邻反馈时间超过
`config.yml` 中的 `maximum_feedback_bracket_s` 时，该条示教不能被接受。

对应的 Diffusion Policy shape 定义见 [`dp_shape_meta_leap_proprio.yml`](dp_shape_meta_leap_proprio.yml)。相机仅服务于在线的人手到 LEAP 映射，采集目录和导出的 Zarr 都不会出现 `camera_0` 或 `depth_0`。

### 一个重要限制

如果示教中人手完全闭合后，16 维目标从头到尾几乎恒定，那么模仿学习主要学到的是“持续保持闭合目标”。从受阻到逐渐闭拢的实际运动，主要仍由电机内部位置环、当前电流限制和糖果接触动力学产生；DP 不会从恒定动作标签中凭空学出新的力控规律。

48 维观察仍然有价值：实测位置、速度和电流能让策略区分“正在受阻”和“已逐渐闭合”。建议训练时使用连续观察历史，例如在 10 Hz 数据上设置 `n_obs_steps: 4` 到 `8`。如果以后希望策略主动改变闭合目标来响应阻力，示教者也必须在不同负载状态下给出有区别的目标动作。

## 3. 安装

在仓库根目录运行，使用已经搭好的 `leaptele` 环境，不需要新建环境：

```powershell
cd D:\Code\autograsp\leap-hand-vision-real-sim-teleop
conda activate leaptele
python -m pip install -r .\requirements.txt
python -m pip install -r .\dp-teleop-collector\requirements.txt
```

如果该环境已经能运行本仓库的 D455/LEAP 遥操，第一条依赖安装可以跳过。采集器使用 Zarr v2；不要升级到 Zarr 3。

## 4. 先用 mock 验证完整数据链

mock 不连接 D455 或真实 LEAP，也不会使能电机：

```powershell
python .\dp-teleop-collector\collect_leap_proprio.py `
  --source mock `
  --leap-device mock `
  --headless `
  --mock-auto-episodes 1 `
  --mock-frames-per-episode 30 `
  --output .\dp-teleop-collector\datasets\mock-leap-grasp
```

验证并导出：

```powershell
python .\dp-teleop-collector\validate_leap_proprio.py `
  .\dp-teleop-collector\datasets\mock-leap-grasp

python .\dp-teleop-collector\export_leap_proprio.py `
  .\dp-teleop-collector\datasets\mock-leap-grasp `
  .\dp-teleop-collector\datasets\mock-leap-grasp.zarr
```

## 5. 采集真实抓糖示教

先在 Dynamixel Wizard 中确认端口能找到 ID 0--15，然后关闭 Wizard。清空手周围空间，并确保可以随时断电或急停。下面的命令只控制 LEAP 手指，不连接 FR3 bridge：

```powershell
python .\dp-teleop-collector\collect_leap_proprio.py `
  --source d455 `
  --leap-device real `
  --leap-port COM4 `
  --enable-leap-torque `
  --output .\dp-teleop-collector\datasets\real-candy-grasp
```

把 `COM4` 换成实际端口。程序会显示相机映射预览，但不会保存画面。

按键：

- `G`：开始一条抓取 episode；
- `SPACE`：结束并接受当前 episode；
- `D`：拒绝当前 episode，保留到 `rejected/` 供排查；
- `Q` 或 `E`：安全退出并按配置关闭真实手扭矩。

推荐每条示教都完整记录以下过程：

1. LEAP 手在糖果中、手指张开且人手稳定可见；
2. 按 `G` 开始记录；
3. 人手逐渐完全闭合，并在机器人手指受阻时继续保持闭合；
4. 由独立的机械臂程序抬起手，采集器继续记录，但不读取或控制机械臂；
5. 保持人手闭合，直到 LEAP 手指在减载后逐渐收紧并稳定；
6. 按 `SPACE` 接受；失败、跟踪中断或糖果掉落时按 `D` 重录。

不要在一条抓取 episode 末尾张开手释放糖果。训练/验证集应按完整 episode 划分，不能把同一轨迹的相邻帧分到两边。

## 6. 验证和导出正式数据

```powershell
python .\dp-teleop-collector\validate_leap_proprio.py `
  .\dp-teleop-collector\datasets\real-candy-grasp

python .\dp-teleop-collector\export_leap_proprio.py `
  .\dp-teleop-collector\datasets\real-candy-grasp `
  .\dp-teleop-collector\datasets\real-candy-grasp.zarr
```

默认只导出 `accepted/`。不要用 rejected 或采样周期异常的轨迹训练正式策略。
导出器还会比较电机型号、Mode、Goal Current、PID、单步限速、motor IDs 和
完整遥操配置哈希；mock/real 或这些动力学设置不同的 episode 不能静默混进同一
个 Zarr，请分别导出。

导出的结构为：

```text
real-candy-grasp.zarr/
  data/
    robot_state         float32 [N,48]
    action              float32 [N,16]
    actual_position     float32 [N,16]
    velocity            float32 [N,16]
    present_current_raw int16   [N,16]
    goal_position       float32 [N,16]
    position_error      float32 [N,16]
    timestamp           float64 [N]
    sample_dt           float64 [N]
    valid               bool    [N]
  meta/
    episode_ends        int64   [E]
  manifest.json
```

训练时只需要把 `data/robot_state` 放进 `shape_meta.obs`，把 `data/action` 作为动作。对三个观察分组分别归一化，尤其不要把电流原始计数与弧度直接共用同一尺度。保持 episode 边界，并使用 `meta/episode_ends` 生成时序窗口。

## 7. 脱离遥操后的职责边界

纯 LEAP 策略推理时只做以下事情：读取 16 个实测关节、计算有限差分速度、读取 16 个 Present Current 原始值，并输出 16 个关节位置目标。执行端仍必须保留与采集时相同的关节限位、post-slew 单步限制、串口失败停机和退出关闭扭矩逻辑。

机械臂何时下降、何时抬升、抬升多高仍由另一个程序负责。由于策略不观察 FR3 位姿或图像，它不会学习机械臂轨迹，也不能判断糖果是否真正被抬离盒子；这正是当前“只学手指抓紧过程”的设计边界。

## 8. 可选的旧 RGB-D / FR3 采集器

旧入口 [`collect_demos.py`](collect_demos.py) 仍保留，用于需要 RGB-D、抓取/释放双任务或 FR3 速度动作的高级实验；配套入口是 [`validate_dataset.py`](validate_dataset.py) 和 [`export_diffusion_policy.py`](export_diffusion_policy.py)，shape 示例见 `dp_shape_meta_leap_only.yml` 与 `dp_shape_meta_fr3_leap.yml`。

它不是本次纯 LEAP 抓糖模型的数据入口。不要把旧采集器导出的图像/FR3 数据与新的 48 维本体感知数据混在同一个 Zarr 或同一次训练中。

Diffusion Policy 的 Zarr/ReplayBuffer 约定可参考：

- [Diffusion Policy ReplayBuffer](https://github.com/real-stanford/diffusion_policy#replaybuffer)
- [官方 ReplayBuffer 实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/replay_buffer.py)
