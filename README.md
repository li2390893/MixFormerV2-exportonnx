# MixFormerV2 (Extended README)

- [MixFormerV2 (Extended README)](#mixformerv2-extended-readme)
  - [Exporting an ONNX Model](#exporting-an-onnx-model)
  - [Running the Video Demo \& Visualization](#running-the-video-demo--visualization)

## Exporting an ONNX Model

Script: `export_onnx.py`. It supports:

- Auto-reading template / search sizes from an experiment config
- Static or dynamic batch (`--batch_mode static|dynamic`)
- Automatic post-export validation + ONNX Runtime inference check

Minimal example:

```bash
python export_onnx.py \
  --checkpoint ./models/MixFormerV2/mixformerv2_small.pth.tar \
  --config_name 224_depth4_mlp1_score \
  --output ./models/mixformerv2_online_small.onnx \
  --batch_mode static \
  --batch_size 1 \
  --opset_version 17
```

Common argument notes:

- `--tracker_name`: Corresponds to `lib/config/<tracker_name>/config.py`, default: `mixformer2_vit_online`
- `--config_name`: Experiment name found at `experiments/<tracker_name>/<config_name>.yaml`
- `--config_path`: Explicit YAML path (overrides `--config_name`)
- `--template_size / --search_size / --online_template_size`: If not set, they are parsed automatically from the config
- `--batch_mode dynamic`: Export with dynamic batch dimension (ensure runtime/backend support)

Post-export checks:

1. The script calls `onnx.checker.check_model`
2. Runs a forward pass with onnxruntime and prints output shapes

Deployment tips:

- If TensorRT reports unsupported ops, try lowering `--opset_version 16` or upgrading TRT.
- You can add `--search_size` to force a fixed size and reduce backend optimization uncertainty.

---

## Running the Video Demo & Visualization

Script: `tracking/video_demo.py`

Features:

- Track a single video with a provided initial box or interactive selection (this script example uses arguments)
- Adjustable online template update interval, search area scale, attention visualization (`vis_attn`), etc.
- Supports saving the tracking result video (`--save_video`) and YOLO annotation format (`--save_yolo`)

![track demo](resources/track_basketball.gif)

Example (a VSCode launch config named `video_demo` may already exist):

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

Argument passing mechanism: any flag of the form `--params__<key>` is parsed into `tracker_params["<key>"]` and passed to the `Tracker` instance. Examples:

- `--params__model`: Model weight path
- `--params__search_area_scale`: Search area scale factor
- `--params__update_interval`: Online update interval
- `--params__vis_attn 1`: Visualize attention maps (tracker must support it)

Optional: Initial bbox

```bash
--optional_box X Y W H
```

Outputs:

- Results are written to `debug/` or the working directory (depends on implementation details).
- With `--save_video`, a video file with overlaid predicted boxes is generated.
