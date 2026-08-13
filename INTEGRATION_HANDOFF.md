# 水杯、robot_edge 与 SAM2+VLM 集成交接

## 1. 当前结论

当前系统不是一套程序直接串到底，而是三段相对独立的链路：

```text
水杯 / 小智
  -> MCP WebSocket
  -> mcp_pipe.py
  -> robot-cup-mcp-bridge/calculator.py 的 MCP 工具
  -> navigation_transport.py
  -> ROS /agent/<robot_id>/navigation_command
  -> 义博 robot_edge

义博巡检 Agent
  -> ROS /agent/<robot_id>/inspection_command
  -> robot_inspection_executor.py
  -> 翊宸 task_frame_recorder 服务
  -> RGB / depth / pose / SAM2 / 点云输出

RGB + SAM2 结果
  -> 本仓库 run_sam2_vlm_task.py
  -> 实例证据图、逐物品 VLM JSON、任务汇总
```

已经验证过的是“水杯 -> MCP -> robot_edge 导航”和“巡检 executor -> task_frame_recorder 采集”。

本仓库的 SAM2+VLM 描述器目前是独立命令行程序，尚未注册为水杯 MCP 工具，也没有直接向义博 Agent 发布 ROS 消息。不要把本仓库描述成已经部署完成的实时闭环。

## 2. 水杯如何连接 MCP

水杯通过 OpenClaw/Xiaozhi 提供的 WebSocket endpoint 连接 MCP：

- `mcp_pipe.py`：WebSocket 与 MCP 子进程 stdin/stdout 之间的桥。
- `MCP_ENDPOINT`：指定当前桥接到哪一个水杯/Agent endpoint；修改后必须重启 `mcp_pipe.py`。
- `calculator.py`：实际的 FastMCP Server，注册导航等工具。
- `Processing request of type CallToolRequest`：证明水杯实际调用了工具；`ListToolsRequest` 只证明工具列表被读取。

独立部署仓库为：

```text
https://github.com/Yukie8848-bit/robot-cup-mcp-bridge
```

建议部署路径为：

```text
/home/gr3-torso-perif/semanticmap/robot-cup-mcp-bridge
```

## 3. MCP 启动方式

执行位置：目标机器人 4 号端口、`gr3-torso-perif` 账户、新终端。

```bash
source /opt/ros/noetic/setup.bash
source /home/gr3-torso-perif/catkin_ws/devel/setup.bash

export ROS_MASTER_URI=http://<ROS_MASTER_HOST>:11311
export ROS_IP=<TORSO_ROS_IP>
unset ROS_HOSTNAME

export ROBOT_NAV_EXECUTION_ENABLED=0
export ROBOT_NAV_ROS_HELPER_PYTHON=/usr/bin/python3
export MCP_ENDPOINT='<水杯对应的 WebSocket endpoint>'

cd /home/gr3-torso-perif/semanticmap
git clone https://github.com/Yukie8848-bit/robot-cup-mcp-bridge.git
cd robot-cup-mcp-bridge
python3 -m pip install -r requirements.txt
cp locations.example.json locations.local.json
# 只在车上填写并复核真实坐标，不提交 locations.local.json

bash ./start_mcp_bridge.sh robot12
# 13号车使用 robot13
```

默认保留 `ROBOT_NAV_EXECUTION_ENABLED=0`。真实移动必须先生成计划、人工核对目标，再明确开启执行并调用确认工具。

启动后检查：

```bash
pgrep -af 'mcp_pipe.py|calculator.py'
grep -n 'def request_turn_off_light\|def plan_navigation' calculator.py
```

## 4. 哪些代码负责与义博 robot_edge 通信

### 水杯 MCP 侧

独立仓库 `robot-cup-mcp-bridge`：

- `calculator.py`
  - `plan_navigation`
  - `get_pending_navigation`
  - `confirm_navigation`
  - `cancel_navigation`
  - `get_live_navigation_status`
- `navigation_adapter.py`
  - 读取地点并构造符合 robot_edge 协议的命令 JSON。
- `navigation_confirmation.py`
  - 保存待确认计划，避免语言请求直接触发移动。
- `navigation_transport.py`
  - 向 ROS command topic 发布 JSON，并读取 status topic。
- `navigation_ros_helper.py`
  - 当 Conda Python 不能直接导入 ROS 时，通过 `/usr/bin/python3` 完成 ROS 发布和订阅。
- `locations.local.json`
  - 导航点位与别名。

导航接口：

```text
command: /agent/robot13/navigation_command
status:  /agent/robot13/navigation_status
heartbeat: /agent/robot13/heartbeat
```

robot12 使用相同结构，将 `robot13` 替换为 `robot12`。

### 义博 robot_edge 侧

使用正式目录 `义博源码/robot_edge/robot_edge/`，不要使用已废弃的 `robot_edge_new`：

- `scripts/robot13_navigation_executor.py`
  - 订阅 navigation command，发布 navigation status/heartbeat，并连接 move_base action server。
- `scripts/robot_inspection_executor.py`
  - 订阅 inspection command，依次导航、旋转四方向并调用翊宸采集服务。
- `config/robot13.yaml`
  - robot13 导航 topic 和 action server 配置。
- `config/robot12_inspection.yaml`
  - robot12 巡检 command/status/heartbeat 配置。

义博 Agent 侧对应：

- `scripts/inspection_cruise_skill.py`：构造并发布巡检命令。
- `scripts/skill_runner.py`：调用导航、巡检和操作 skill。
- `config/robot_skills.json`：声明 `inspection_cruise`。
- `config/robot13_inspection_routes.json`：巡检路线。

巡检接口：

```text
command: /agent/<robot_id>/inspection_command
status: /agent/<robot_id>/inspection_status
heartbeat: /agent/<robot_id>/inspection_heartbeat
```

## 5. 翊宸采集与 SAM2 服务部署位置

执行位置：目标机器人 9 号端口/VNC、`gr3-torso-vision` 账户、新终端。

```text
源码：/home/gr3-torso-vision/catkin_ws/src/save_video
输出：/home/gr3-torso-vision/videos/<task_id>/
```

Orbbec 与采集服务启动：

```bash
source /opt/ros/noetic/setup.bash
source /home/gr3-torso-vision/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://<ROS_MASTER_HOST>:11311
export ROS_IP=<VISION_ROS_IP>
unset ROS_HOSTNAME

sudo systemctl start auto_start_orbbec_driver.service
cd /home/gr3-torso-vision/catkin_ws
./src/save_video/start_task_frame_recorder.sh
```

检查：

```bash
rosnode list | grep -E 'orbbec/camera|task_frame_recorder|semantic_pointcloud_pipeline'
rosservice list | grep task_frame_recorder
timeout 10s rostopic echo -n 1 /orbbec/color/image_raw/header
timeout 10s rostopic echo -n 1 /orbbec/depth/image_raw/header
```

## 6. 本仓库部署与启动

公开仓库：

```text
https://github.com/Yukie8848-bit/vlm-semantic-observation-demo
```

建议部署到视觉账户，因为真实 RGB、SAM2 和任务目录都位于该账户，避免跨账户读取权限问题：

```text
/home/gr3-torso-vision/semanticmap/vlm-semantic-observation-demo
```

启动一次离线任务描述：

```bash
cd /home/gr3-torso-vision/semanticmap
git clone https://github.com/Yukie8848-bit/vlm-semantic-observation-demo.git
cd vlm-semantic-observation-demo
python3 -m pip install -r requirements.txt
cp .env.example .env
# 在 .env 中配置 API_KEY、BASE_URL、MODEL_NAME

python3 scripts/run_sam2_vlm_task.py \
  --task-id inspection_003 \
  --rgb-dir <真实RGB目录> \
  --sam2-dir <SAM2 JSON和labels目录> \
  --output-dir outputs/inspection_003_vlm
```

详细输入输出见 `SAM2_VLM_TASK.md`。

## 7. 后续如何真正接入水杯

最小集成方案建议由翊宸继续完成：

1. 在视觉账户启动一个轻量 ROS service，例如 `/semantic_observation/describe_task`。
2. service 输入至少包含 `task_id`、RGB 目录、SAM2 目录；内部调用 `run_sam2_vlm_task.py`。
3. service 返回 `success`、`task_summary.json` 路径和简短物品摘要。
4. 在 `robot-cup-mcp-bridge/calculator.py` 新增 MCP 工具 `describe_inspection_task(task_id)`，调用该 ROS service。
5. 水杯先调用描述工具，MCP 将结构化结果返回给水杯；后续如需导航或操作，仍经过独立计划和人工确认。

建议使用 ROS service，而不是让 MCP 直接读取 `/home/gr3-torso-vision/videos`：此前不同账户对任务清单存在读取权限差异，直接共享文件路径不够稳定。

完成以上五步之前，只能说“VLM 描述器可以处理 SAM2 结果”，不能说“水杯已经能实时调用 VLM 描述器”。

## 8. 交付验收清单

- [ ] 机器人端确认当前实际运行的是独立仓库中的 `calculator.py`。
- [ ] `MCP_ENDPOINT` 对应目标水杯，重启桥后出现 `CallToolRequest`。
- [ ] `calculator.py` 中确认口语任务工具和人工确认逻辑已同步。
- [ ] robot_edge heartbeat、command/status topic 均在线。
- [ ] Orbbec RGB/depth 有持续 header，task_frame_recorder 服务在线。
- [ ] SAM2+VLM 仓库部署到视觉账户并完成一次真实 API 小样测试。
- [ ] 新增 ROS description service 和 MCP wrapper 后，再做水杯端端到端验收。
