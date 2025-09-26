#!/usr/bin/env python3
"""
Export MixFormerV2 model to ONNX format
"""

import os
import sys
import argparse
import importlib
from copy import deepcopy
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import onnx
import onnxruntime

# Add project path to sys.path
prj_path = os.path.join(os.path.dirname(__file__), ".")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.models.mixformer2_vit import build_mixformer2_vit_online


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


def _get_cfg_value(cfg_obj: Any, path: Sequence[str], fallback: Any = None) -> Any:
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
            message = f"{module_path} must expose a default cfg or provide a config"
            raise AttributeError(message)
        cfg = deepcopy(base_cfg)
        print("Loaded default configuration (no experiment overrides).")

    return cfg


def export_model_to_onnx(
    cfg,
    checkpoint_path,
    output_path,
    template_size,
    search_size,
    online_template_size=None,
    opset_version=17,
    batch_size=1,
    dynamic_batch=False,
):
    """Export MixFormerV2 model to ONNX using dynamic configuration values."""

    # Load model configuration
    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Build model
    model = build_mixformer2_vit_online(cfg, train=False)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["net"], strict=True)
    print("Checkpoint loaded successfully!")

    # Set model to evaluation mode
    model.eval()
    export_module = MixFormerOnnxWrapper(model, softmax=True)
    export_module.eval()

    # Create dummy inputs for tracing
    template_height, template_width = _normalize_hw(template_size)
    search_height, search_width = _normalize_hw(search_size)
    if online_template_size is None:
        online_template_source = template_size
    else:
        online_template_source = online_template_size
    online_template_height, online_template_width = _normalize_hw(
        online_template_source
    )

    # Template input (for initialization)
    template_input = torch.randn(
        batch_size,
        3,
        template_height,
        template_width,
    )

    # Online template input (can be same as template for single template mode)
    online_template_input = torch.randn(
        batch_size, 3, online_template_height, online_template_width
    )

    # Search region input
    search_input = torch.randn(
        batch_size,
        3,
        search_height,
        search_width,
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_module = export_module.to(device)
    template_input = template_input.to(device)
    online_template_input = online_template_input.to(device)
    search_input = search_input.to(device)

    print(f"Using device: {device}")
    print("Input shapes:")
    print(f"  Template: {template_input.shape}")
    print(f"  Online Template: {online_template_input.shape}")
    print(f"  Search: {search_input.shape}")

    # Export the model to ONNX
    print("Exporting model to ONNX...")

    # Define input names and dynamic axes
    input_names = ["template", "online_template", "search"]
    output_names = ["pred_boxes", "pred_scores"]

    dynamic_axes = {
        "template": {0: "batch_size"},
        "online_template": {0: "batch_size"},
        "search": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"},
        "pred_scores": {0: "batch_size"},
    }

    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "verbose": True,
        "export_params": True,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = dynamic_axes

    # Export with torch.onnx.export
    torch.onnx.export(
        export_module,
        (template_input, online_template_input, search_input),
        output_path,
        **export_kwargs,
    )

    print(f"ONNX model saved to: {output_path}")

    # Verify the exported model
    print("Verifying exported ONNX model...")

    # Load the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")

    # Test inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(output_path)

    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        "template": template_input.cpu().numpy(),
        "online_template": online_template_input.cpu().numpy(),
        "search": search_input.cpu().numpy(),
    }

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)

    print("ONNX Runtime inference successful!")
    print("Output shapes:")
    boxes_shape = getattr(ort_outputs[0], "shape", "unknown")
    scores_shape = getattr(ort_outputs[1], "shape", "unknown")
    print(f"  pred_boxes: {boxes_shape}")
    print(f"  pred_scores: {scores_shape}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export MixFormerV2 model to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mixformer2_vit.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="mixformer2_vit_online",
        help="Tracker name used to locate configuration modules",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="224_depth4_mlp1_score",
        help="Experiment YAML name (without .yaml extension)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a YAML config file (overrides --config_name)",
    )
    parser.add_argument(
        "--search_size",
        type=int,
        default=None,
        help="Override search region size for dummy inputs",
    )
    parser.add_argument(
        "--template_size",
        type=int,
        default=None,
        help="Override template size for dummy inputs",
    )
    parser.add_argument(
        "--online_template_size",
        type=int,
        default=None,
        help="Override online template size (defaults to template size)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Deprecated alias for --search_size",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version to use when exporting",
    )
    parser.add_argument(
        "--batch_mode",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help=(
            "Choose whether to export with a static or dynamic batch dimension"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to embed into the exported model (default: 1)",
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist!")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg = load_config(
        tracker_name=args.tracker_name,
        parameter_name=args.config_name,
        config_path=args.config_path,
        project_root=prj_path,
    )

    template_size = args.template_size
    if template_size is None:
        template_size = _get_cfg_value(
            cfg,
            ("TEST", "TEMPLATE_SIZE"),
            fallback=_get_cfg_value(cfg, ("DATA", "TEMPLATE", "SIZE")),
        )
    if template_size is None:
        raise ValueError(
            "Template size not found in configuration; specify --template_size"
        )

    search_size = args.search_size
    if search_size is None and args.input_size is not None:
        search_size = args.input_size
    if search_size is None:
        search_size = _get_cfg_value(
            cfg,
            ("TEST", "SEARCH_SIZE"),
            fallback=_get_cfg_value(cfg, ("DATA", "SEARCH", "SIZE")),
        )
    if search_size is None:
        raise ValueError(
            "Search size not found in configuration; specify --search_size"
        )

    online_template_size = args.online_template_size or template_size

    print(f"Tracker name: {args.tracker_name}")
    if args.config_path:
        resolved_config = _ensure_absolute(args.config_path, prj_path)
        print(f"Config path: {resolved_config}")
    else:
        print(f"Config name: {args.config_name}")
    print(f"Template size: {template_size}")
    print(f"Online template size: {online_template_size}")
    print(f"Search size: {search_size}")
    print(f"Opset version: {args.opset_version}")
    print(f"Batch mode: {args.batch_mode}")
    print(f"Batch size: {args.batch_size}")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")

    dynamic_batch = args.batch_mode == "dynamic"

    try:
        success = export_model_to_onnx(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            template_size=template_size,
            search_size=search_size,
            online_template_size=online_template_size,
            opset_version=args.opset_version,
            batch_size=args.batch_size,
            dynamic_batch=dynamic_batch,
        )

        if success:
            print("ONNX export completed successfully!")
        else:
            print("ONNX export failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
