# MixFormerV2 (Extended README)

- [MixFormerV2 (Extended README)](#mixformerv2-extended-readme)
  - [导出 ONNX 模型](#导出-onnx-模型)
  - [视频 Demo 运行与可视化](#视频-demo-运行与可视化)

## 导出 ONNX 模型

脚本：`export_onnx.py`。支持：

- 基于实验配置自动读取模板 / 搜索尺寸
- 静态或动态 batch（`--batch_mode static|dynamic`）
- 自动进行导出后合法性检查 + ONNX Runtime 推理验证

最小示例：

```bash
python export_onnx.py \
  --checkpoint ./models/MixFormerV2/mixformerv2_small.pth.tar \
  --config_name 224_depth4_mlp1_score \
  --output ./models/mixformerv2_online_small.onnx \
  --batch_mode static \
  --batch_size 1 \
  --opset_version 17
```

常用参数说明：

- `--tracker_name`：对应 `lib/config/<tracker_name>/config.py`，默认 `mixformer2_vit_online`
- `--config_name`：`experiments/<tracker_name>/<config_name>.yaml` 中的实验配置名
- `--config_path`：显式指定 YAML（覆盖 `--config_name`）
- `--template_size / --search_size / --online_template_size`：若不设，将从配置里自动解析
- `--batch_mode dynamic`：导出带动态 batch 维度（部署需注意 runtime 支持）

导出后验证：

1. 脚本会自动 `onnx.checker.check_model`
2. 使用 onnxruntime 做一次前向并打印输出 shape

部署提示：

- 若后端是 TensorRT 且遇到算子不支持，可尝试降低 `--opset_version 16` 或升级 TRT 版本。
- 可通过添加 `--search_size` 统一固定尺寸，减少后端优化不确定性。

---

## 视频 Demo 运行与可视化

脚本：`tracking/video_demo.py`

功能：

- 对单个视频进行跟踪，可选初始框或手动交互（当前脚本示例使用传参）
- 可调节在线模板更新频率、搜索区域缩放、可视化注意力 (vis_attn) 等
- 支持保存跟踪结果视频 (`--save_video`) 与 YOLO 标注格式 (`--save_yolo`)

示例（VSCode Launch 已提供一个配置 `video_demo`）：

```bash
python tracking/video_demo.py \
  mixformer2_vit_online \
  224_depth4_mlp1_score \
  /path/to/video.mp4 \
  --params__model models/MixFormerV2/mixformerv2_small.pth.tar \
  --params__search_area_scale 5.0 \
  --zoomin \
  --save_video
```

传参机制说明：所有形如 `--params__<key>` 的参数会被解析为内部 `tracker_params["<key>"]`，最终传入 `Tracker` 实例。例如：

- `--params__model`：模型权重路径
- `--params__search_area_scale`：搜索区域缩放因子
- `--params__update_interval`：在线更新间隔
- `--params__vis_attn 1`：可视化注意力图（需 tracker 支持）

可选：初始 bbox

```bash
--optional_box X Y W H
```

输出：

- 结果可在 `debug/` 或运行目录下生成（依具体实现）。
- 使用 `--save_video` 会生成叠加预测框的视频文件。