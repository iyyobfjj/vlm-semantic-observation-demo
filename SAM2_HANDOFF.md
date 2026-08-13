# SAM2 实例分割描述系统交接说明

## 交付范围

该工具读取已经生成的 RGB 和 SAM2 结果，在本地离线完成：

1. 校验 RGB、SAM2 JSON 和标签图是否逐帧对应。
2. 过滤面积为零的记录及天花板、地面、地毯等背景类别。
3. 输出物品类别、track、边界框、中心点和图像二维方位。
4. 生成带实例边界、中文名称和 track 编号的效果图。
5. 生成逐帧中文描述、整批物品清单和可直接播报的文本。
6. 生成 `acceptance_report.json` 供程序或人工验收。

该工具不调用 ROS 或网络服务，也不输出地图坐标、真实距离或三维位置。

## 环境准备

```powershell
conda activate gr
cd <vlm-semantic-observation-demo 仓库路径>
python -m pip install -r requirements.txt
```

## 一键运行

当前工作区的 RGB、SAM2 和输出目录已经配置为默认值：

```powershell
python scripts\run_sam2_pipeline.py
```

处理其他数据时显式指定目录：

```powershell
python scripts\run_sam2_pipeline.py `
  --rgb-dir <RGB目录> `
  --sam2-dir <SAM2目录> `
  --output-dir <输出目录>
```

## 输入要求

每个帧 ID 需要以下三个核心文件：

```text
rgb/<frame_id>.png
sam2/<frame_id>.json
sam2/<frame_id>_labels.png
```

当 RGB 文件是从 Linux 复制来的符号链接文本时，工具会尝试使用：

```text
sam2/<frame_id>_overlay.png
```

回退情况会记录在 `base_source_counts`，不会被报告成真实 RGB 输入。

## 输出目录

```text
outputs/sam2_description/
  acceptance_report.json
  scene_summary.json
  visualization_manifest.json
  speech_summary.json
  summary.txt
  frames/                 # 逐帧结构化观测
  overlays/               # 带中文标签的实例分割效果图
  descriptions/frames/    # 逐帧中文描述
```

二维位置字段使用九宫格描述，例如“左上方”“中央”“右下方”。这些位置仅表示图像中的方位。

## 当前样本验收基线

当前 30 帧样本的预期结果：

```text
frame_count=30
visible_observation_count=28
unique_track_count=8
rendered_frame_count=30
description_frame_count=30
mask_metadata_alignment_ok=true
```

物品清单为：3 个储物柜、2 个开关、1 个冰箱、1 个微波炉、1 个垃圾桶。

当前本地 30 个 RGB 文件是 Linux 路径引用，因此验收报告中的：

```text
rgb_verified_frame_count=0
base_source_counts={"sam2_overlay": 30}
```

这不影响现有 SAM2 二维实例结果，但若后续需要从原始 RGB 重新生成掩膜叠加图，应复制真实图片文件。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖输入匹配、背景过滤、二维位置、track 汇总、效果图、中文描述和一键管线。
