# VLM Semantic Observation Demo

要把 SAM2 分割物体生成 VLM 描述并关联 PCD 点云，请直接看
[SAM2 物体描述与点云关联教程](POINTCLOUD_DESCRIPTION_GUIDE.md)。
该教程也包含与原整图 VLM 描述合成统一 JSON 的命令。

接口字段和实现边界见 [SAM2_VLM_TASK.md](SAM2_VLM_TASK.md)。原
`run_sam2_pipeline.py` 仍是无网络、无 API 费用的确定性离线基线。

水杯、MCP、义博 `robot_edge`、翊宸采集服务和本仓库之间的当前连接边界、部署位置及启动方式见 [INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md)。

本仓库包含两条相互独立的离线流程：

1. **SAM2 实例结果转语义描述**：读取已有 RGB、SAM2 元数据和标签图，输出物品清单、图像二维位置、标注图和中文播报文本。这是当前交付的主要入口。
2. **VLM API 语义观测 Demo**：调用兼容 API，将普通场景图片转换为结构化语义观测并写入 SQLite。

## SAM2 实例结果转语义描述

### 适用场景

用于把翊宸侧已经生成的 SAM2 实例分割结果整理成义博侧巡检界面或播报模块容易消费的结果。该流程完全离线，不会调用 ROS、机器人服务或外部 VLM API。

输入内容：

```text
rgb/<frame_id>.png
sam2/<frame_id>.json
sam2/<frame_id>_labels.png
```

每个实例会输出类别、track ID、边界框、中心点、掩码面积和图像二维方位。天花板、地面、地毯等背景类别会被过滤。

### 快速运行

```powershell
conda activate gr
cd <vlm-semantic-observation-demo 仓库路径>
python -m pip install -r requirements.txt

python scripts\run_sam2_pipeline.py `
  --rgb-dir <RGB目录> `
  --sam2-dir <SAM2目录> `
  --output-dir outputs\sam2_description
```

在当前开发目录结构中，也可以直接使用脚本内的默认目录：

```powershell
python scripts\run_sam2_pipeline.py
```

### 输出内容

```text
outputs/sam2_description/
  acceptance_report.json       # 整批处理是否成功及产物路径
  scene_summary.json            # 去重后的物品实例和整批统计
  speech_summary.json           # 中文总结及可直接播报的文本
  summary.txt                   # 便于人工查看的文本摘要
  visualization_manifest.json  # 原始帧、标注图和描述文件的对应关系
  frames/                       # 逐帧结构化实例观测 JSON
  overlays/                     # 带实例框、名称和 track ID 的效果图
  descriptions/frames/         # 逐帧中文描述
```

当前 30 帧样例的验收结果为：

```json
{
  "success": true,
  "frame_count": 30,
  "visible_observation_count": 28,
  "unique_track_count": 8,
  "rendered_frame_count": 30,
  "description_frame_count": 30,
  "mask_metadata_alignment_ok": true
}
```

对应的中文播报文本为：

```text
视觉巡检完成，识别到3个储物柜、2个开关、1个冰箱、1个微波炉和1个垃圾桶。
```

这里的“左侧”“中央”“右下方”等位置只表示物品在图像中的二维方位，不代表地图坐标、真实距离或三维位置。当前流程也不负责物品间空间关系推理、机器人导航和机械臂操作。

完整的输入约定、字段说明、样例基线和测试命令见 [SAM2_HANDOFF.md](SAM2_HANDOFF.md)。

## 原有 VLM API 语义观测 Demo

这个 demo 用于验证机器人关灯巡检/公司场景服务机器人的第一阶段最小闭环：

本地图像输入 -> 调用公司 VLM API -> 输出结构化 JSON -> 保存本地 JSON/SQLite -> 支持简单语义查询 -> 返回原图路径和语义观测结果。

它只负责 VLM 语义观测模块，重点说明机器人视角能看到什么、当前灯光状况如何、灯是否可能开启、开关是否可见、开关状态是否可判断，以及是否需要关灯。不做本地大 VLM 部署、3DGS、SLAM、机械臂控制、实时视频流或复杂 Web 前端。

## 一键准备环境

如果你已经安装了 Miniforge/conda，推荐直接运行：

```bash
cd vlm_semantic_observation_demo
bash scripts/setup_env.sh --backend conda
```

脚本会创建默认环境 `vlm-semobs`，安装 `requirements.txt`，并在缺少 `.env` 时从 `.env.example` 复制一份。

以后进入环境：

```bash
conda activate vlm-semobs
cd vlm_semantic_observation_demo
```

也可以自定义环境名和 Python 版本：

```bash
bash scripts/setup_env.sh --backend conda --name vlm-semobs --python 3.10
```

如果没有 conda，脚本会在 `--backend auto` 下尝试使用 `.venv`：

```bash
bash scripts/setup_env.sh
```

## 手动安装依赖

```bash
cd vlm_semantic_observation_demo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

如果系统提示缺少 `ensurepip` 或 `python3-venv`，先安装对应的 Python venv 支持包，或直接在已有 Python 环境中执行 `python3 -m pip install -r requirements.txt`。

## 配置 API

```bash
cp .env.example .env
```

编辑 `.env`：

- `VLM_API_STYLE=openai`：用于 OpenAI-compatible 的公司 VLM API。
- `API_KEY`：公司 API key。
- `BASE_URL`：OpenAI-compatible base URL，例如 `https://api.example.com/v1`。
- `MODEL_NAME`：公司提供的 VLM 模型名。
- `FORCE_IPV4=true`：强制 OpenAI-compatible 请求使用 IPv4，避免不可达 IPv6 地址导致长时间连接超时。
- `VLM_API_STYLE=requests`：用于普通 HTTP POST 接口，并配置 `VLM_ENDPOINT`。
- `MAX_OUTPUT_TOKENS`：单次模型调用的最大输出 token 数，默认 `500`。

`requests` 模式默认发送：

```json
{
  "model": "MODEL_NAME",
  "prompt": "系统提示词 + 用户提示词",
  "image": "data:image/jpeg;base64,...",
  "temperature": 0,
  "max_tokens": 500
}
```

如果公司接口字段不同，只需要改 `src/vlm_client.py` 里的 `_call_requests_api`。

### OpenAI 兼容纯文本测速

在排查图片理解速度前，可以先用同一个 API key、Base URL 和模型发送一次最短纯文本请求：

```bash
python scripts/benchmark_text_api.py
```

脚本默认关闭思考模式和 SDK 内部重试，仅要求模型返回 `OK`，并输出请求耗时、token usage 和 reasoning 状态。连续测试三次可使用：

```bash
python scripts/benchmark_text_api.py --repeat 3
```

临时测试另一个模型而不修改 `.env`：

```bash
python scripts/benchmark_text_api.py --model qwen3.6-flash
```

测量流式请求的首 Token 时间：

```bash
python scripts/benchmark_text_api.py --stream
```

使用同一个 key 测试 Anthropic 兼容端点：

```bash
python scripts/benchmark_anthropic_text_api.py
```

## 放入图片

把 20 到 50 张室内关键帧图片放入：

```text
data/images/
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

## 批量调用 VLM API

```bash
python3 scripts/run_vlm_api.py
```

可选参数：

```bash
python3 scripts/run_vlm_api.py --area-hint 会议室
python3 scripts/run_vlm_api.py --overwrite
python3 scripts/run_vlm_api.py --model qwen3.6-flash
```

如果使用仓库内公开示例数据 `data/images/switch_01/`，可以这样跑：

```bash
python3 scripts/run_vlm_api.py --image-dir data/images/switch_01 --output-dir outputs/json_switch_01 --area-hint 开关面板
```

对应构建 SQLite：

```bash
python3 scripts/build_sqlite.py --json-dir outputs/json_switch_01 --db-path outputs/switch_01.sqlite
```

对应查询：

```bash
python3 scripts/search_demo.py --db-path outputs/switch_01.sqlite --switch-visible
python3 scripts/search_demo.py --db-path outputs/switch_01.sqlite --need-action
python3 scripts/search_demo.py --db-path outputs/switch_01.sqlite --task light_off
```

脚本会：

- 遍历 `data/images/`；
- 跳过已生成的 `outputs/json/img_xxx.json`；
- 调用 VLM API；
- 尝试解析 JSON；
- 用 Pydantic 校验 schema；
- 输出机器人视角说明字段 `robot_view`；
- 输出关灯巡检字段 `light_inspection`；
- 失败时保存 `outputs/json/img_xxx.failed.json`，其中包含错误信息和 raw response。

## 构建 SQLite

```bash
python3 scripts/build_sqlite.py
```

默认生成：

```text
outputs/semantic_observations.sqlite
```

SQLite 表包括：

- `images`
- `robot_view`
- `objects`
- `light_inspection`
- `switches`
- `abnormalities`
- `relations`
- `uncertainty`

## 查询

```bash
python3 scripts/search_demo.py --keyword 线缆
python3 scripts/search_demo.py --risk medium
python3 scripts/search_demo.py --area 会议室
python3 scripts/search_demo.py --abnormal
python3 scripts/search_demo.py --uncertain
python3 scripts/search_demo.py --light-on
python3 scripts/search_demo.py --switch-visible
python3 scripts/search_demo.py --need-action
python3 scripts/search_demo.py --task light_off
```

查询结果会输出匹配到的：

- `image_path`
- `area_type`
- `scene_summary`
- `robot_view`
- `light_inspection`
- `objects`
- `abnormalities`
- `uncertainty`

其中 `image_path` 是后续回看证据图的入口。

## 结构化 JSON

核心输出不是普通 caption，而是用于语义地图存储的关灯巡检语义观测：

```json
{
  "image_id": "img_001",
  "image_path": "data/images/img_001.jpg",
  "timestamp": null,
  "area_hint": null,
  "scene_summary": "画面整体描述",
  "area_type": "会议室/调试区/工具区/货架区/走廊/未知",
  "robot_view": {
    "visible_summary": "机器人视角能看到左侧墙面开关、右侧室内区域、天花板灯和地面纸箱",
    "visible_area": "门口附近",
    "key_visible_elements": ["墙面开关", "天花板灯", "纸箱", "门框"],
    "lighting_condition_description": "画面整体较亮，右侧区域可见天花板灯发光，同时可能存在窗外自然光",
    "occlusions_or_blind_spots": ["开关拨动方向不清晰", "右侧区域部分被门框遮挡"],
    "image_quality": "清晰",
    "robot_view_limitation": "只能根据当前单帧图像判断，无法确认开关控制哪一路灯"
  },
  "objects": [
    {
      "name": "墙面开关",
      "category": "light_switch",
      "location_description": "门口右侧墙面",
      "state": "疑似开启",
      "attributes": ["固定", "可操作", "关灯任务相关"],
      "inspection_relevance": "关灯巡检任务相关",
      "risk_level": "none",
      "suggested_action": "靠近开关进一步确认或执行关灯",
      "confidence": 0.72
    }
  ],
  "light_inspection": {
    "room_lighting_state": "on",
    "ambient_light_level": "bright",
    "visible_light_sources": ["天花板灯"],
    "switch_visibility": "visible",
    "switches": [
      {
        "visible": true,
        "location_description": "门口右侧墙面",
        "state": "uncertain",
        "evidence": "开关面板可见，但拨动方向不清晰",
        "confidence": 0.65
      }
    ],
    "need_turn_off": "uncertain",
    "evidence": "房间明亮且可见天花板灯发光，但无法排除自然光影响",
    "suggested_action": "靠近开关或结合时间/自然光信息进一步确认",
    "confidence": 0.7
  },
  "spatial_relations": ["墙面开关位于门口右侧"],
  "abnormalities": [],
  "uncertainty": ["无法确认亮度是否来自自然光", "无法确认开关拨动方向"],
  "raw_model_response": null
}
```

如果只需要让机器人泛化描述“当前看见了什么”，可以参考 `src/scene_description_prompt.py`。它输出紧凑的英文机器人视角场景 JSON，包括 `scene_brief`、`overall_lighting`、`items` 和 `uncertainty`。其中 `items` 最多记录 6 个主要物品，并按 `item1`、`item2` 记录物品可能是什么、形状、图中位置、当前视角下是否可操作和置信度。`operable` 只表示图像中交互部位清晰可见且未被明显阻挡，不代表导航、机械臂可达性或实际操作一定成功。

也可以通过同一个批处理脚本选择 prompt 模式：

```bash
# 关灯巡检，默认模式
python3 scripts/run_vlm_api.py --prompt-mode light_off --image-dir data/images/switch_01 --output-dir outputs/json_switch_01 --area-hint 开关面板

# 通用机器人视角场景描述
python3 scripts/run_vlm_api.py --prompt-mode scene_description --image-dir data/images/switch_01 --output-dir outputs/json_scene_description --area-hint 开关面板
```

通用场景 JSON 使用独立脚本写入 SQLite，避免和关灯巡检 schema 混用：

```bash
python3 scripts/build_scene_sqlite.py \
  --json-dir outputs/json_scene_description \
  --db-path outputs/scene_observations.sqlite
```

也可以一次导入多批结果。若不同目录中存在相同的 `image_id`，后导入的记录会覆盖先导入的记录：

```bash
python3 scripts/build_scene_sqlite.py \
  --json-dir outputs/json_distance_test_01 \
  --json-dir outputs/json_refrigerator_01_en \
  --db-path outputs/all_scenes.sqlite
```

通用场景数据库包含：

- `scene_images`：图片路径、场景摘要和整体灯光；
- `scene_items`：物品名称、形状、图中位置和置信度；
- `scene_uncertainty`：无法可靠确认的信息。

按物品名称查询：

```bash
python3 scripts/search_scene.py --db-path outputs/all_scenes.sqlite --item refrigerator
python3 scripts/search_scene.py --db-path outputs/all_scenes.sqlite --item 冰箱
python3 scripts/search_scene.py --db-path outputs/all_scenes.sqlite --item switch --min-confidence 0.8
```

`--item` 使用名称子串匹配；英文匹配不区分大小写。查询结果返回原图路径、场景摘要、灯光情况、主要物品和不确定性。省略 `--item` 时会列出数据库中的场景记录。

## 当前限制

- 不是实时系统；
- 不部署本地 VLM；
- 不做 3D 建图；
- 不做大型向量数据库；
- 不做 grounding 模型评测；
- 不保证单张图片一定能可靠判断物理开关状态，必须把看不清的开关、自然光干扰、曝光问题写入 `uncertainty`；
- 只验证关灯巡检与通用场景描述的 VLM 语义观测、结构化存储和简单检索小闭环。

## 验收标准

- 20 张图片能批量生成 JSON；
- JSON 大部分能被 Pydantic 校验通过；
- SQLite 能正常构建；
- 能按对象、区域、风险、异常、不确定性、灯光状态、开关可见性、是否需要关灯查询；
- 查询结果返回原始图片路径，作为证据图。
