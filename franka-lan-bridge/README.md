# Franka FR3 LAN Bridge

这个项目让两台同一局域网内的电脑协作：

- **FR3 控制电脑（Linux、无需显卡）**运行 `franka_server.py`，本地导入并调用现有的 [`FrankaController`](https://github.com/Zyyyyoyo/Franka_control/blob/main/utils/franka_controller.py)；
- **摄像头/遥操电脑**运行 `franka_client.py` 或导入 `FrankaBridgeClient`，发送受限的高层运动目标并实时读取机械臂状态。

Franka FCI 的 1 kHz 实时循环仍然只存在于 FR3 控制电脑与机器人之间。局域网桥工作在 20–30 Hz，不把 1 kHz 控制、急停或安全判断搬到 Windows/摄像头电脑。Franka 官方说明 FCI 实时控制要求本地控制工作站维持 1 kHz 循环和实时内核，因此这个边界不能颠倒。

## 已实现的接口

- 20 Hz 完整状态流：7 关节位置、速度、力矩、外力矩、接触/碰撞，末端位姿、速度、外力和错误状态；
- 单一控制权租约：可同时有多个观察客户端，但同一时刻只有一个控制客户端；
- `global` / `local` 连续笛卡尔速度；
- 服务端速度、单步位移、XYZ 工作空间和 dynamics factor 二次限幅；
- 相对位移和全局相对/绝对位置接口（默认关闭）；
- 软件停止；任何通过认证的观察客户端都能停止并撤销当前控制权；
- 可选错误恢复（默认关闭）；
- token 认证和客户端 IP/CIDR 白名单；
- 假机器人模式，可在不连接 FR3 时验证两台电脑的网络。

## 为什么采用这种结构

`FrankaController` 的连续速度命令本身只有很短的生命周期。本项目在机器人电脑上以固定频率重复刷新最新目标，同时设置第二层网络看门狗：

- Franky 命令生命周期默认 `150 ms`；
- 客户端速度目标超过 `200 ms` 没更新，机器人侧主动停止；
- 控制租约超过 `1000 ms` 没有速度或心跳，停止并释放控制权；
- 普通心跳只能维持控制租约，**不能延长旧速度目标**；
- TCP/WebSocket 断开时，机器人侧立即释放控制并停止；
- 机器人侧进程异常时，最后一条 Franky 短生命周期命令仍会超时归零。

这个接口适合遥操末端速度、发送小幅笛卡尔动作和读取状态，不适合从远端发送 1 kHz 力矩、关节阻抗或原始 FCI 数据。

## 1. 准备两台电脑的 IP

FR3 Linux 电脑：

```bash
hostname -I
ip -4 addr
```

Windows 遥操电脑：

```powershell
ipconfig
```

下面假设：

```text
FR3 控制电脑：192.168.1.20
Windows 遥操电脑：192.168.1.10
服务端口：8765
```

请把示例地址替换为实际地址，并尽量使用有线局域网和固定 IP。

## 2. 把项目放到 FR3 控制电脑

可以从 Windows PowerShell 复制：

```powershell
scp -r D:\Code\autograsp\franka-lan-bridge asus@192.168.1.20:/home/asus/
```

在 FR3 控制电脑安装网络依赖。这里沿用现有 `zyy` 环境，不创建新环境：

```bash
cd /home/asus/franka-lan-bridge
/home/asus/miniconda3/envs/zyy/bin/python -m pip install -r requirements.txt
cp server_config.example.json server_config.json
```

编辑 `server_config.json`：

- `controller_root` 指向包含 `utils/franka_controller.py` 的仓库根目录，按现有项目通常是 `/home/asus/zyy`；
- `robot_host` 填 FR3 的 FCI 地址；
- 把 `allowed_client_cidrs` 中的 `192.168.1.10/32` 改成 Windows 电脑的准确 IP；
- 第一次联调保持 `allow_one_shot_motion=false` 和 `allow_error_recovery=false`；
- 根据真实工作台设置 `workspace_min_m`、`workspace_max_m`，不要直接沿用示例范围。

## 3. 设置共享密钥

在任一电脑生成一次随机 token：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

把生成值分别放入两台电脑的环境变量，不要写进 Git：

Linux 服务端：

```bash
export FRANKA_BRIDGE_TOKEN='替换为生成的随机值'
```

Windows 客户端：

```powershell
$env:FRANKA_BRIDGE_TOKEN='替换为同一个随机值'
```

本协议默认使用明文 `ws://`，token 不能防止局域网抓包。只在可信、隔离的实验室有线网络使用，不要把端口映射到公网。

## 4. 先运行假机器人服务

FR3 电脑：

```bash
cd /home/asus/franka-lan-bridge
/home/asus/miniconda3/envs/zyy/bin/python franka_server.py \
  --config server_config.json \
  --dry-run
```

`--dry-run` 不导入 `franky`，也不会连接或移动真实机器人。

Windows 电脑安装同一网络依赖：

```powershell
cd D:\Code\autograsp\franka-lan-bridge
D:\Environment\Anaconda\envs\leaptele\python.exe -m pip install -r requirements.txt
```

检查端口：

```powershell
Test-NetConnection -ComputerName 192.168.1.20 -Port 8765
```

读取一次假状态：

```powershell
D:\Environment\Anaconda\envs\leaptele\python.exe franka_client.py `
  --uri ws://192.168.1.20:8765 state
```

持续读取 JSON 状态：

```powershell
D:\Environment\Anaconda\envs\leaptele\python.exe franka_client.py `
  --uri ws://192.168.1.20:8765 monitor
```

Linux 端确认监听：

```bash
ss -ltnp | grep 8765
```

如果启用了 UFW，只放行 Windows 电脑的准确 IP：

```bash
sudo ufw allow from 192.168.1.10 to any port 8765 proto tcp
```

## 5. 连接真实 FR3

开始前确认：

1. Franka Desk 已进入 Execution 模式；
2. FCI 已激活；
3. FR3 控制电脑到机器人 FCI 网口保持原来的低延迟连接；
4. 急停可用，机器人附近无人，工作空间已清空；
5. `server_config.json` 的速度和 XYZ 工作空间已经按实机收紧；
6. 已先通过 `--dry-run` 验证 token、白名单、防火墙和双向状态。

去掉 `--dry-run` 启动真实服务：

```bash
/home/asus/miniconda3/envs/zyy/bin/python franka_server.py \
  --config server_config.json
```

先从 Windows 读取状态，不发运动：

```powershell
python franka_client.py --uri ws://192.168.1.20:8765 state
```

任何已认证客户端都可以停止：

```powershell
python franka_client.py --uri ws://192.168.1.20:8765 stop
```

第一次速度测试建议只用 `0.005 m/s`、持续 1 秒，并把手放在实体急停附近：

```powershell
python franka_client.py --uri ws://192.168.1.20:8765 velocity `
  --frame global `
  --linear 0.005 0 0 `
  --angular 0 0 0 `
  --duration 1 `
  --rate 30
```

命令结束、程序异常或按 `Ctrl+C` 时，客户端会请求停止；机器人侧看门狗仍是最终保障。

## 6. 一次性笛卡尔运动

默认 `allow_one_shot_motion=false`。完成速度遥操联调并实测工作空间后，才能在服务端改为 `true`。

局部坐标相对移动 5 mm：

```powershell
python franka_client.py --uri ws://192.168.1.20:8765 move-relative `
  --xyz 0.005 0 0 `
  --dynamics-factor 0.1
```

Base 坐标相对移动 5 mm：

```powershell
python franka_client.py --uri ws://192.168.1.20:8765 move-global `
  --xyz 0 0 0.005 `
  --dynamics-factor 0.1
```

加 `--absolute` 才会把 XYZ 解释为 Base 下的绝对位置。服务端会检查：

- 位移范数不超过 `max_relative_displacement_m`；
- 最终位置位于 `workspace_min_m` 和 `workspace_max_m` 内；
- dynamics factor 不超过 `max_motion_dynamics_factor`；
- 客户端持续持有控制租约，断线立即停止一次性动作。

## 7. 标定四个点并按键顺序运动

这两个程序都运行在 Windows 客户端。第一步，先在一个 PowerShell 窗口建立 SSH 隧道，并保持这个窗口运行：

```powershell
ssh -N `
  -L 8765:127.0.0.1:8765 `
  -p 6855 `
  用户名@10.15.89.229
```

然后在另一个 PowerShell 中设置与服务端相同的 token：

```powershell
$env:FRANKA_BRIDGE_TOKEN='与服务端完全相同的随机密钥'
```

第二步，先读取一次 FR3 状态，确认 SSH 隧道、身份认证和服务端都已正常连接：

```powershell
python franka_client.py `
  --uri ws://127.0.0.1:8765 `
  state
```

只有这条命令能正常返回机器人状态后，才继续运行下面的标定或顺序运动程序。

### 标定程序

标定程序只读取机器人状态，不申请控制权，也不会主动移动机械臂。先通过手动引导或已有的安全方式把末端放到目标位置，再按一次 `SPACE`：

```powershell
python franka_calibrate_points.py `
  --uri ws://127.0.0.1:8765 `
  --output calibrated_points.json
```

按键：

- `SPACE`：记录当前末端 Base 坐标 XYZ；
- `U` 或退格键：撤销上一个点；
- `Q`、`E` 或 `Esc`：取消，不写文件。

依次记录 `P1`、`P2`、`P3`、`P4` 后自动保存。重新标定已有文件时加 `--overwrite`：

```powershell
python franka_calibrate_points.py `
  --uri ws://127.0.0.1:8765 `
  --output calibrated_points.json `
  --overwrite
```

保存文件同时记录了各点的旋转矩阵用于复核，但现有 `FrankaController.move_global()` 只接收 XYZ，所以回放时保持机械臂开始运动时的当前末端旋转。

### 顺序运动程序

真实运动前必须在 FR3 服务端的 `server_config.json` 中：

1. 根据实际工作台收紧 `workspace_min_m` 和 `workspace_max_m`；
2. 将 `allow_one_shot_motion` 改为 `true`；
3. 重启 `franka_server.py`。

然后运行：

```powershell
python franka_next_point.py `
  --uri ws://127.0.0.1:8765 `
  --points calibrated_points.json `
  --dynamics-factor 0.05
```

按键：

- 每按一次 `SPACE`，依次规划到 `P1 → P2 → P3 → P4`；
- 运动过程中按 `E`、`Q` 或 `Esc`，立即发送软件停止并退出；
- 到达每个点后检查实际 XYZ，默认误差必须不超过 `5 mm`；
- P4 完成后程序释放控制权并退出。

从其他点开始：

```powershell
python franka_next_point.py `
  --uri ws://127.0.0.1:8765 `
  --points calibrated_points.json `
  --start-index 3
```

加 `--loop` 可在 P4 后回到 P1。标定文件默认被 `.gitignore` 排除，不会意外提交真实机器人位姿。

## 8. 在遥操程序中调用客户端库

```python
import asyncio

from franka_bridge.client import FrankaBridgeClient


async def main():
    async with FrankaBridgeClient("ws://192.168.1.20:8765") as fr3:
        # 不申请控制权也可以读取状态。
        state = await fr3.next_state(timeout_s=2.0)
        print(state["robot"]["end_effector"]["position"])

        await fr3.acquire_control()
        try:
            # 遥操循环建议 30 Hz；每隔若干帧等待一次 ACK。
            for frame in range(60):
                await fr3.send_velocity(
                    linear=(0.005, 0.0, 0.0),
                    angular=(0.0, 0.0, 0.0),
                    frame="global",
                    wait_ack=frame % 10 == 0,
                )
                await asyncio.sleep(1.0 / 30.0)
        finally:
            await fr3.stop()
            await fr3.release_control()


asyncio.run(main())
```

后续把视觉腕部映射接到 FR3 时，只需将每帧的末端位置增量转换为受限速度，再调用 `send_velocity()`；LEAP Hand 的 16 关节控制仍可以保持独立。

## 9. 本机测试

```powershell
D:\Environment\Anaconda\envs\autograsp\python.exe -m unittest discover -s tests -v
```

测试只使用假机器人和 `127.0.0.1`，覆盖速度超时停止、租约、控制权冲突、序号防重放、工作空间限制、观察端停止和完整客户端/服务端回环通信，不连接真实 FR3。

## 目录

- `franka_server.py`：FR3 控制电脑入口；
- `franka_client.py`：Windows 通用命令行入口；
- `franka_calibrate_points.py`：按 SPACE 记录四个末端点；
- `franka_next_point.py`：按 SPACE 顺序运动到下一个点；
- `franka_bridge/server.py`：认证、状态流和请求分发；
- `franka_bridge/runtime.py`：机器人侧租约、看门狗和安全状态机；
- `franka_bridge/client.py`：可嵌入遥操程序的异步客户端；
- `franka_bridge/waypoints.py`：四点文件格式和工作空间验证；
- `franka_bridge/terminal_keys.py`：无额外依赖的非阻塞单键输入；
- `server_config.example.json`：安全配置模板；
- `tests/`：无硬件单元和回环测试。
