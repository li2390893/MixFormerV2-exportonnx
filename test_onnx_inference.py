#!/usr/bin/env python3
"""
Test ONNX inference for MixFormerV2 model, save inputs, and compare errors
"""

import os
import sys
import argparse
import importlib
from copy import deepcopy
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import onnxruntime
import numpy as np
from lib.models.mixformer2_vit import build_mixformer2_vit_online

# Add project path to sys.path
prj_path = os.path.join(os.path.dirname(__file__), ".")
if prj_path not in sys.path:
    sys.path.append(prj_path)


class MixFormerOnnxWrapper(nn.Module):
    def __init__(self, model, softmax=True):
        super().__init__()
        self.model = model
        self.softmax = softmax

    def forward(self, template, online_template, search):
        outputs = self.model(
            template,
            online_template,
            search,
            softmax=self.softmax,
            run_score_head=True,
        )
        boxes = outputs["pred_boxes"]
        scores = outputs["pred_scores"]
        scores = torch.sigmoid(scores)
        return boxes, scores


def _ensure_absolute(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def _normalize_hw(size: Union[int, Iterable[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"Invalid size specification: {size}")


def _get_cfg_value(
    cfg_obj: Any, path: Sequence[str], fallback: Any = None
) -> Any:
    node = cfg_obj
    for key in path:
        try:
            if isinstance(node, dict):
                node = node[key]
            else:
                node = getattr(node, key)
        except (KeyError, AttributeError, TypeError):
            return fallback
    return node


def load_config(
    tracker_name: str,
    parameter_name: Optional[str],
    config_path: Optional[str],
    project_root: str,
) -> Any:
    """Load tracker configuration similar to Tracker.get_parameters."""

    module_path = f"lib.config.{tracker_name}.config"
    config_module = importlib.import_module(module_path)

    resolved_config_path: Optional[str] = None
    if config_path:
        resolved_config_path = _ensure_absolute(config_path, project_root)
    elif parameter_name:
        resolved_config_path = os.path.join(
            project_root,
            "experiments",
            tracker_name,
            f"{parameter_name}.yaml",
        )

    if resolved_config_path:
        if not os.path.exists(resolved_config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {resolved_config_path}"
            )
        if not hasattr(config_module, "update_new_config_from_file"):
            raise AttributeError(
                "Config module missing update_new_config_from_file function"
            )
        cfg = config_module.update_new_config_from_file(resolved_config_path)
        print(f"Loaded configuration from: {resolved_config_path}")
    else:
        base_cfg = getattr(config_module, "cfg", None)
        if base_cfg is None:
            message = (
                f"{module_path} must expose a default cfg or provide a config"
            )
            raise AttributeError(message)
        cfg = deepcopy(base_cfg)
        print("Loaded default configuration (no experiment overrides).")

    return cfg


def save_inputs_to_npy(inputs_dir, template_input, online_template_input, search_input):
    """Save input tensors as .npy files"""
    os.makedirs(inputs_dir, exist_ok=True)
    np.save(os.path.join(inputs_dir, "template_input.npy"), template_input.cpu().numpy())
    np.save(os.path.join(inputs_dir, "online_template_input.npy"), online_template_input.cpu().numpy())
    np.save(os.path.join(inputs_dir, "search_input.npy"), search_input.cpu().numpy())
    print(f"Inputs saved to {inputs_dir}")


def compare_outputs(pytorch_outputs, onnx_outputs, threshold=1e-4):
    """Compare PyTorch and ONNX outputs, return error metrics"""
    pytorch_boxes, pytorch_scores = pytorch_outputs
    onnx_boxes, onnx_scores = onnx_outputs

    boxes_diff = np.abs(pytorch_boxes - onnx_boxes)
    scores_diff = np.abs(pytorch_scores - onnx_scores)

    boxes_max_error = np.max(boxes_diff)
    boxes_mean_error = np.mean(boxes_diff)
    scores_max_error = np.max(scores_diff)
    scores_mean_error = np.mean(scores_diff)

    boxes_close = np.allclose(pytorch_boxes, onnx_boxes, rtol=threshold, atol=threshold)
    scores_close = np.allclose(pytorch_scores, onnx_scores, rtol=threshold, atol=threshold)

    return {
        "boxes_max_error": boxes_max_error,
        "boxes_mean_error": boxes_mean_error,
        "scores_max_error": scores_max_error,
        "scores_mean_error": scores_mean_error,
        "boxes_close": boxes_close,
        "scores_close": scores_close,
    }


def test_onnx_inference(
    cfg,
    checkpoint_path,
    onnx_path,
    template_size,
    search_size,
    online_template_size=None,
    save_inputs_dir=None,
    error_threshold=1e-4,
    batch_size=1,
    template_npy=None,
    online_template_npy=None,
    search_npy=None,
):
    """Test ONNX inference, save inputs, and compare with PyTorch"""

    # Build and load PyTorch model
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    export_module = MixFormerOnnxWrapper(model, softmax=True)
    export_module.eval()

    # Create dummy inputs
    template_height, template_width = _normalize_hw(template_size)
    search_height, search_width = _normalize_hw(search_size)
    if online_template_size is None:
        online_template_source = template_size
    else:
        online_template_source = online_template_size
    online_template_height, online_template_width = _normalize_hw(online_template_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_module = export_module.to(device)

    if template_npy:
        template_input = torch.from_numpy(np.load(template_npy)).to(device)
    else:
        template_input = torch.randn(batch_size, 3, template_height, template_width).to(device)

    if online_template_npy:
        online_template_input = torch.from_numpy(np.load(online_template_npy)).to(device)
    else:
        online_template_input = torch.randn(batch_size, 3, online_template_height, online_template_width).to(device)

    if search_npy:
        search_input = torch.from_numpy(np.load(search_npy)).to(device)
    else:
        search_input = torch.randn(batch_size, 3, search_height, search_width).to(device)

    # Save inputs if directory provided
    if save_inputs_dir:
        save_inputs_to_npy(save_inputs_dir, template_input, online_template_input, search_input)

    # PyTorch inference
    with torch.no_grad():
        pytorch_outputs = export_module(template_input, online_template_input, search_input)
        pytorch_boxes = pytorch_outputs[0].cpu().numpy()
        pytorch_scores = pytorch_outputs[1].cpu().numpy()

    # ONNX inference
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {
        "template": template_input.cpu().numpy(),
        "online_template": online_template_input.cpu().numpy(),
        "search": search_input.cpu().numpy(),
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_boxes, onnx_scores = ort_outputs

    print("PyTorch outputs:")
    print(f"  Boxes: {pytorch_boxes}")
    print(f"  Scores: {pytorch_scores}")
    print("ONNX outputs:")
    print(f"  Boxes: {onnx_boxes}")
    print(f"  Scores: {onnx_scores}")

    # Compare outputs
    errors = compare_outputs((pytorch_boxes, pytorch_scores), (onnx_boxes, onnx_scores), error_threshold)

    print("Error comparison:")
    print(f"  Boxes - Max error: {errors['boxes_max_error']:.6f}, Mean error: {errors['boxes_mean_error']:.6f}, Close: {errors['boxes_close']}")
    print(f"  Scores - Max error: {errors['scores_max_error']:.6f}, Mean error: {errors['scores_mean_error']:.6f}, Close: {errors['scores_close']}")

    if not (errors['boxes_close'] and errors['scores_close']):
        print(f"Warning: Outputs differ beyond threshold {error_threshold}")
        return False
    else:
        print("ONNX inference matches PyTorch within threshold")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test ONNX inference for MixFormerV2")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--tracker_name", type=str, default="mixformer2_vit_online", help="Tracker name")
    parser.add_argument("--config_name", type=str, default="224_depth4_mlp1_score", help="Config name")
    parser.add_argument("--config_path", type=str, default=None, help="Optional config path")
    parser.add_argument("--template_size", type=int, default=None, help="Template size")
    parser.add_argument("--search_size", type=int, default=None, help="Search size")
    parser.add_argument("--online_template_size", type=int, default=None, help="Online template size")
    parser.add_argument("--save_inputs_dir", type=str, default=None, help="Directory to save inputs")
    parser.add_argument("--error_threshold", type=float, default=1e-4, help="Error threshold for comparison")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--template_npy", type=str, default=None, help="Path to template input .npy file")
    parser.add_argument("--online_template_npy", type=str, default=None, help="Path to online template input .npy file")
    parser.add_argument("--search_npy", type=str, default=None, help="Path to search input .npy file")

    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"Error: ONNX file {args.onnx_path} does not exist!")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist!")
        sys.exit(1)

    cfg = load_config(args.tracker_name, args.config_name, args.config_path, prj_path)

    template_size = args.template_size
    if template_size is None:
        template_size = _get_cfg_value(cfg, ("TEST", "TEMPLATE_SIZE"), fallback=_get_cfg_value(cfg, ("DATA", "TEMPLATE", "SIZE")))
    if template_size is None:
        raise ValueError("Template size not found; specify --template_size")

    search_size = args.search_size
    if search_size is None:
        search_size = _get_cfg_value(cfg, ("TEST", "SEARCH_SIZE"), fallback=_get_cfg_value(cfg, ("DATA", "SEARCH", "SIZE")))
    if search_size is None:
        raise ValueError("Search size not found; specify --search_size")

    online_template_size = args.online_template_size or template_size

    try:
        success = test_onnx_inference(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            onnx_path=args.onnx_path,
            template_size=template_size,
            search_size=search_size,
            online_template_size=online_template_size,
            save_inputs_dir=args.save_inputs_dir,
            error_threshold=args.error_threshold,
            batch_size=args.batch_size,
            template_npy=args.template_npy,
            online_template_npy=args.online_template_npy,
            search_npy=args.search_npy,
        )
        if success:
            print("Test passed!")
        else:
            print("Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
典型使用方法：

1. 导出 ONNX 模型：
   python export_onnx.py --checkpoint checkpoints/train/mixformer2_vit/teacher_288_depth12/xxx.pth --output mixformer2_vit.onnx --tracker_name mixformer2_vit_online --config_name 288_depth8_score

2. 测试推理并保存输入：
   python test_onnx_inference.py --onnx_path mixformer2_vit.onnx --checkpoint checkpoints/train/mixformer2_vit/teacher_288_depth12/xxx.pth --save_inputs_dir ./test_inputs --tracker_name mixformer2_vit_online --config_name 288_depth8_score

3. 对比误差：脚本会自动输出误差指标，如果超出阈值则警告。
"""