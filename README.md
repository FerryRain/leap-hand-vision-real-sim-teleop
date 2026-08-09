# LEAP Hand Vision Teleoperation: Real Hand + MuJoCo Mirror

这个仓库只做一件事：摄像头识别人手的 21 个 MediaPipe 关键点，将手指姿态映射为 LEAP Hand 的 16 个关节目标，并把同一组安全限幅后的目标同时发送到：

- 真实 LEAP Hand；
- MuJoCo 中固定腕部的 LEAP Hand 镜像。

没有机械臂、物体抓取、腕部移动或末端位置控制。真实手和仿真手始终共享同一组 16 维目标。

## 运行环境

推荐 Windows 10/11、Python 3.10，并使用已经搭好的 `leaptele` Conda 环境。先进入本仓库目录，然后安装依赖：

```powershell
cd D:\Code\autograsp\leap-hand-vision-real-sim-teleop
D:\Environment\Anaconda\envs\leaptele\python.exe -m pip install -r requirements.txt
```

如果该环境的实际路径不同，可以先查看：

```powershell
conda env list
```

后续示例中的 `python` 均指安装了 `requirements.txt` 的同一个 Python。

## 先做无硬件安全测试

下面的命令会打开摄像头预览和 MuJoCo 窗口，但不会连接电机：

```powershell
python teleop.py --device mock
```

如果只想做自动化冒烟测试，不打开摄像头和窗口：

```powershell
python teleop.py --source mock --device mock --headless --no-preview --no-realtime --duration 1
```

## 查看串口号

插入 LEAP Hand 后运行：

```powershell
python -m serial.tools.list_ports -v
```

也可以在“设备管理器 → 端口 (COM 和 LPT)”中查看。选择 Dynamixel Wizard 能识别电机 ID `0` 到 `15` 的同一个端口。启动遥操前必须关闭 Dynamixel Wizard，避免它占用串口。

## 同时控制真实手和 MuJoCo

先让手指远离夹点和物体，确认电源、电机 ID、端口以及急停方式无误。假设端口为 `COM4`：

```powershell
python teleop.py --device real --port COM4 --enable-torque
```

指定其他摄像头：

```powershell
python teleop.py --device real --port COM4 --enable-torque --camera-index 1
```

程序先打开 MuJoCo 和摄像头，然后才连接真实手。连接时会读取并保持真实手当前姿态；人手连续稳定识别 8 帧后才开始跟随。每次发送还会受到 `config.yml` 中的关节范围和平滑/步长限制。

控制键：

- `SPACE`：暂停或继续，暂停时保持当前关节目标；
- `L`：重新加载 `config.yml` 中的映射参数；
- `Q` 或 `E`：停止并退出；
- 关闭 MuJoCo 窗口：停止并退出。

正常退出、摄像头/串口错误或关闭 MuJoCo 窗口时，程序都会尝试关闭真实手扭矩。断电或软件退出后，机械手可能因失去保持力而突然运动或掉落手中物体，请提前托住并清空周围空间。

## 调整人手到 LEAP Hand 的映射

编辑根目录的 `config.yml`，运行时按 `L` 即可热重载。常用参数：

- `mapping.joint_open_rad`：完全张手时的 16 维 LEAP 目标；
- `mapping.joint_closed_rad`：完全闭手时的 16 维 LEAP 目标；
- `mapping.long_human_*_deg`：食指、中指、无名指的人手弯曲角标定；
- `mapping.thumb_human_*_deg`：拇指三段弯曲角标定；
- `mapping.thumb_opposition_*`：拇指对掌映射；
- `tuning.finger_gain`：食指、中指、无名指、拇指的独立闭合增益；
- `tuning.landmark_smoothing_alpha`：关键点滤波，越小越稳、越大响应越快；
- `control.joint_deadband_rad`：忽略小幅摄像头抖动；
- `hardware.maximum_step_rad`：真实电机每次命令的最大变化量。

关节顺序固定为：

```text
IF [mcp, rot, pip, dip]
MF [mcp, rot, pip, dip]
RF [mcp, rot, pip, dip]
TH [cmc, axl, mcp, ipl]
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

运行日志写入 `logs/session.json`，包含视觉目标、真实手实测关节、仿真实测关节、误差和退出原因。`logs/` 默认不会提交到 Git。

## 硬件说明

默认配置面向 LEAP Hand v1：Dynamixel Protocol 2.0、4 Mbps、ID `0-15`。真实电机模式必须同时给出 `--device real`、`--port COMx` 和 `--enable-torque`，缺少任一项都会拒绝使能。请先在 `--device mock` 模式确认映射方向和范围；本仓库的自动测试不会替代真实硬件上的低速、空载验证。

## 致谢与许可证

人手映射思路参考 Julianxng 的 LEAP Hand 遥操实现，整体视觉遥操结构参考 fzhang327 的 MuJoCo 项目。详细来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。仓库代码使用 MIT License；LEAP Hand MuJoCo 模型的许可证保留在 `models/leap_hand/LICENSE`。
