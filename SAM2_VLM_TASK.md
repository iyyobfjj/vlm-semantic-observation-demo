# SAM2 + VLM 实例描述闭环

## 目标与边界

该入口把已经生成的 RGB、SAM2 元数据和实例标签图整理为实例级语义描述：

1. 按 `(sam2_label, track_id)` 合并同一物品的跨帧观测。
2. 每个 track 选择掩码面积最大的代表帧，只调用一次 VLM。
3. 为 VLM 生成三联证据图：原场景目标高亮、掩码隔离目标、目标局部裁剪。
4. 将 VLM 的可见外观描述与 SAM2 的 track、二维位置和证据路径合并。
5. 输出逐物品 JSON、任务汇总和验收报告。

该入口不调用 ROS、不控制机器人、不修改 `save_video`，也不根据单张图推断地图坐标、机械臂可达性或操作成功率。

## 输入

每一帧必须具有严格对应的三个文件：

```text
rgb/<frame_id>.png
sam2/<frame_id>.json
sam2/<frame_id>_labels.png
```

用于 VLM 的 RGB 必须是真实、可解码的图片。仅包含 Linux 路径文字的占位 PNG 仍可用于原离线解析流程，但不能作为本入口的视觉证据。

天花板、地面、地毯、墙面等背景类别会被过滤，不调用 VLM。

## 配置 VLM

从示例创建 `.env`，并配置公司可用的兼容接口：

```powershell
Copy-Item .env.example .env
```

至少需要：

```text
VLM_API_STYLE=openai
API_KEY=<key>
BASE_URL=<兼容接口地址>
MODEL_NAME=<视觉模型名>
```

## 运行

```powershell
python scripts\run_sam2_vlm_task.py `
  --task-id inspection_003 `
  --rgb-dir <RGB目录> `
  --sam2-dir <SAM2目录> `
  --output-dir outputs\inspection_003_vlm
```

同一输出目录默认不允许覆盖。确认要重跑时增加 `--overwrite`，或使用新的任务输出目录。

## 输出

```text
outputs/inspection_003_vlm/
  acceptance_report.json
  task_summary.json
  evidence/
    <label>_track_<id>.png
  objects/
    <label>_track_<id>.json
  failures/
    <label>_track_<id>.json
```

每个成功对象 JSON 包含：

- SAM2 标签、track ID、代表帧和出现过的帧；
- 图像二维位置、归一化中心和面积占比；
- RGB、SAM2 JSON、标签图、证据图路径；
- VLM 返回的物品名称、类别、可见外观、状态、交互部位、遮挡、置信度和不确定性；
- 可选的显式点云路径。

VLM 返回无法解析或不符合 Schema 时，不生成伪造对象结果，而是在 `failures/` 保存错误、原始返回和证据图路径。

## 可选点云绑定

SAM2 的 `track_id` 不等于 ROS 语义点云的 `object_id`。因此程序不会按文件名或编号猜测关联，只接受人工或上游程序确认过的显式映射。

示例 `pointcloud_map.json`：

```json
{
  "switch:7": "pointclouds/wall_switch_0001.pcd",
  "refrigerator:2": "pointclouds/refrigerator_0001.pcd"
}
```

相对路径以映射文件所在目录为基准，目标文件必须存在。运行时增加：

```powershell
--pointcloud-map pointcloud_map.json
```

## 测试

测试使用假的 VLM 客户端，不访问网络，也不会产生 API 费用：

```powershell
python -m unittest discover -s tests -v
```

验收重点：同一 track 只调用一次 VLM；代表帧选择正确；背景不调用 VLM；失败结果可追溯；点云只能通过显式映射绑定。
