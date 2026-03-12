#!/usr/bin/env python3
"""
Test ONNX inference for MixFormerV2 model, save inputs, compare errors, and support running ONNX tracking on image sequences.

Features added:
- Read initial box from text (format: x0 y0 x1 y1 w h cx cy class_id) and initial image
- Read search image(s) (single image or directory) and run ONNX inference for each frame
- Draw predicted box and score on the frames and save to output directory
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
import cv2 as cv
from lib.test.tracker.tracker_utils import PreprocessorX_onnx, Preprocessor_wo_mask
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
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
    init_box_txt: Optional[str] = None,
    init_image: Optional[str] = None,
    search_image: Optional[str] = None,
    output_dir: Optional[str] = None,
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

    # Default PyTorch tensors (may be overridden by image-mode below)
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

    # If image-mode is requested (init image + bbox + single search image) then perform inference on images
    def read_init_box(txt_path: str):
        """Read first line of the bbox text file. Only x0, y0, x1, y1 are required.
        Additional fields (w, h, cx, cy, class_id) are optional; w,h,cx,cy will be computed
        from x0,x1,y0,y1. If class_id is provided as the last token, it will be returned as int.
        Supported formats:
          x0 y0 x1 y1
          x0 y0 x1 y1 class_id
          x0 y0 x1 y1 w h cx cy class_id  (still supported but w/h/cx/cy are ignored)
        """
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
        ss = line.split()
        if len(ss) < 4:
            raise ValueError("Invalid init_box format. Expected at least 'x0 y0 x1 y1'")
        x0, y0, x1, y1 = map(float, ss[:4])
        # Normalize coordinates: ensure x <= x1, y <= y1
        x_min = min(x0, x1)
        y_min = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        # compute center coordinates (not used directly here but kept for completeness)
        _cx = x_min + 0.5 * w
        _cy = y_min + 0.5 * h
        x = x_min
        y = y_min
        class_id = 0
        # If class id is present as last token, parse it
        if len(ss) >= 5:
            try:
                class_id = int(ss[-1])
            except Exception:
                class_id = 0
        return [x, y, w, h], class_id

    def map_box_back(pred_box, prev_state, resize_factor, search_sz):
        """Map pred_box returned by model back to image coordinates.
        pred_box: iterable [cx, cy, w, h] (in search patch coordinates)
        prev_state: [x, y, w, h] previous state's box
        resize_factor: float (search patch resize factor returned by sample_target)
        search_sz: int search crop size (pixels)
        returns: [x, y, w, h]
        """
        cx_prev = prev_state[0] + 0.5 * prev_state[2]
        cy_prev = prev_state[1] + 0.5 * prev_state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * search_sz / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def draw_box(image, box, score=None, color=(0, 0, 255)):
        # box: [x,y,w,h]
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
        if score is not None:
            cv.putText(image, f"{score:.3f}", (int(x1), int(y1)-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    # If image-mode is requested, override .npy/random inputs with image-derived inputs for both PyTorch and ONNX
    if init_box_txt and init_image and search_image:
        preproc_x = PreprocessorX_onnx()
        preproc_torch = Preprocessor_wo_mask()
        init_box_parsed, class_id = read_init_box(init_box_txt)
        init_img_bgr = cv.imread(init_image)
        if init_img_bgr is None:
            raise FileNotFoundError(f"Init image not found: {init_image}")
        init_img = cv.cvtColor(init_img_bgr, cv.COLOR_BGR2RGB)
        # template
        z_res = sample_target(init_img, init_box_parsed, _get_cfg_value(cfg, ("TEST", "TEMPLATE_FACTOR"), fallback=_get_cfg_value(cfg, ("DATA", "TEMPLATE", "FACTOR"), 2.0)), output_sz=template_size)
        z_patch_arr = z_res[0]
        z_amask_arr = z_res[2] if len(z_res) > 2 else z_res[1]
        img_template_np, _ = preproc_x.process(z_patch_arr, np.asarray(z_amask_arr))
        torch_template = preproc_torch.process(z_patch_arr)
        # online template (same as template by default)
        img_online_template_np = img_template_np
        torch_online_template = torch_template
        # single search image
        search_img_bgr = cv.imread(search_image)
        if search_img_bgr is None:
            raise FileNotFoundError(f"Search image not found: {search_image}")
        search_img = cv.cvtColor(search_img_bgr, cv.COLOR_BGR2RGB)
        x_res = sample_target(search_img, init_box_parsed, _get_cfg_value(cfg, ("TEST", "SEARCH_FACTOR"), fallback=_get_cfg_value(cfg, ("DATA", "SEARCH", "FACTOR"), 4.5)), output_sz=search_size)
        x_patch_arr = x_res[0]
        resize_factor = x_res[1]
        x_amask_arr = x_res[2] if len(x_res) > 2 else x_res[1]
        img_search_np, _ = preproc_x.process(x_patch_arr, np.asarray(x_amask_arr))
        torch_search = preproc_torch.process(x_patch_arr)
        # set PyTorch inputs
        template_input = torch_template.to(device)
        online_template_input = torch_online_template.to(device) if isinstance(torch_online_template, torch.Tensor) else torch.tensor(torch_online_template).to(device)
        search_input = torch_search.to(device)
        # set ONNX inputs
        ort_img_template = img_template_np.astype(np.float32)
        ort_img_online_template = img_online_template_np.astype(np.float32)
        ort_img_search = img_search_np.astype(np.float32)
    with torch.no_grad():
        pytorch_outputs = export_module(template_input, online_template_input, search_input)
        pytorch_boxes = pytorch_outputs[0].cpu().numpy()
        pytorch_scores = pytorch_outputs[1].cpu().numpy()

    # ONNX inference
    ort_session = onnxruntime.InferenceSession(onnx_path)
    if init_box_txt and init_image and search_image:
        ort_inputs = {
            "template": ort_img_template,
            "online_template": ort_img_online_template,
            "search": ort_img_search,
        }
    else:
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
        # continue but still return False
        result_ok = False
    else:
        print("ONNX inference matches PyTorch within threshold")
        result_ok = True

    # If requested, run image-based sequence of detections using the ONNX model and init box
    if init_box_txt and init_image and search_image:
        # Prepare preprocessors: ONNX (numpy) and PyTorch (torch cuda)
        preproc_x = PreprocessorX_onnx()
        preproc_torch = Preprocessor_wo_mask()
        # Read init box and initial image
        init_box, class_id = read_init_box(init_box_txt)
        # read init image (cv2 returns BGR; convert to RGB)
        init_img_bgr = cv.imread(init_image)
        if init_img_bgr is None:
            raise FileNotFoundError(f"Init image not found: {init_image}")
        init_img = cv.cvtColor(init_img_bgr, cv.COLOR_BGR2RGB)

        # Create template and online_template inputs once
        z_res = sample_target(init_img, init_box, _get_cfg_value(cfg, ("TEST", "TEMPLATE_FACTOR"), fallback=_get_cfg_value(cfg, ("DATA", "TEMPLATE", "FACTOR"), 2.0)), output_sz=template_size)
        z_patch_arr = z_res[0]
        z_amask_arr = z_res[2] if len(z_res) > 2 else z_res[1]
        img_template_input, _ = preproc_x.process(z_patch_arr, np.asarray(z_amask_arr))
        torch_template_input = preproc_torch.process(z_patch_arr)
        # online_template: use same as template
        img_online_template_input = img_template_input
        torch_online_template_input = torch_template_input

        # Build list of search images
        # search_image is a single image
        search_images = [search_image]

        # Ensure output_dir exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        prev_state = init_box
        for idx, frame_path in enumerate(search_images):
            search_img_bgr = cv.imread(frame_path)
            if search_img_bgr is None:
                print(f"Warning: search image not found: {frame_path}")
                continue
            search_img = cv.cvtColor(search_img_bgr, cv.COLOR_BGR2RGB)

            x_res = sample_target(search_img, prev_state, _get_cfg_value(cfg, ("TEST", "SEARCH_FACTOR"), fallback=_get_cfg_value(cfg, ("DATA", "SEARCH", "FACTOR"), 4.5)), output_sz=search_size)
            x_patch_arr = x_res[0]
            resize_factor = x_res[1]
            x_amask_arr = x_res[2] if len(x_res) > 2 else x_res[1]
            img_search_input, _ = preproc_x.process(x_patch_arr, np.asarray(x_amask_arr))
            torch_search_input = preproc_torch.process(x_patch_arr)

            # Run ONNX inference
            ort_inputs_single = {
                "template": img_template_input.astype(np.float32),
                "online_template": img_online_template_input.astype(np.float32),
                "search": img_search_input.astype(np.float32),
            }
            ort_outs = ort_session.run(None, ort_inputs_single)
            boxes_np, scores_np = ort_outs

            # Handle boxes shape (B, N, 4) or (B, 4N)
            # We'll compute mean across queries to get one box
            if boxes_np.ndim == 3:
                # (1, N, 4)
                mean_box = boxes_np.mean(axis=1).reshape(-1)
            elif boxes_np.ndim == 2 and boxes_np.shape[1] % 4 == 0:
                # (1, 4N) flattened
                N = boxes_np.shape[1] // 4
                boxes_np_reshaped = boxes_np.reshape(1, N, 4)
                mean_box = boxes_np_reshaped.mean(axis=1).reshape(-1)
            else:
                raise ValueError("Unexpected shape for pred_boxes from ONNX: {}".format(boxes_np.shape))

            # The model returns coordinates relative to the search patch; follow the tracker mapping
            # Convert mean_box to CPU np array
            pred_box = (mean_box * search_size / resize_factor).tolist()
            pred_score = float(np.mean(scores_np)) if scores_np.size else 0.0
            mapped_box = map_box_back(pred_box, prev_state, resize_factor, search_size)
            # Clip using same clip_box as tracker (margin=10)
            H, W = search_img.shape[:2]
            mapped_box = clip_box(mapped_box, H, W, margin=10)

            # Print result
            print(f"Frame {idx} - pred_box: {mapped_box}, score: {pred_score:.4f}")

            # Draw and save result image
            draw_img = draw_box(search_img_bgr.copy(), mapped_box, score=pred_score)
            if output_dir:
                save_path = os.path.join(output_dir, f"{idx:04d}.jpg")
                cv.imwrite(save_path, draw_img)

            # Track online template update logic (from mixformer2_vit_online.track)
            # Initialize online variables if not set
            if 'max_pred_score' not in locals():
                max_pred_score = -1.0
                online_max_template = torch_template_input
                online_forget_id = 0
                online_template = torch_online_template_input
                online_size = 1
                update_interval = 1
                if isinstance(_get_cfg_value(cfg, ("TEST", "ONLINE_SIZES"), fallback=None), (list, tuple)):
                    online_size = _get_cfg_value(cfg, ("TEST", "ONLINE_SIZES"), fallback=[1])[0]
                elif isinstance(_get_cfg_value(cfg, ("TEST", "ONLINE_SIZES"), fallback=None), dict):
                    val = _get_cfg_value(cfg, ("TEST", "ONLINE_SIZES"), fallback=None)
                    try:
                        online_size = list(val.values())[0][0]
                    except Exception:
                        online_size = 1
                update_intervals = _get_cfg_value(cfg, ("TEST", "UPDATE_INTERVALS"), fallback=None)
                if isinstance(update_intervals, (list, tuple)):
                    update_interval = update_intervals[0]
                elif isinstance(update_intervals, dict):
                    try:
                        update_interval = list(update_intervals.values())[0][0]
                    except Exception:
                        update_interval = _get_cfg_value(cfg, ("DATA", "MAX_SAMPLE_INTERVAL"), fallback=1)

            # Update online template if the prediction meets condition
            if pred_score > 0.5 and pred_score > max_pred_score:
                # extract template from image using mapped_box
                z_res2 = sample_target(search_img, mapped_box, _get_cfg_value(cfg, ("TEST", "TEMPLATE_FACTOR"), fallback=_get_cfg_value(cfg, ("DATA", "TEMPLATE", "FACTOR"), 2.0)), output_sz=template_size)
                z_patch_arr2 = z_res2[0]
                z_amask_arr2 = z_res2[2] if len(z_res2) > 2 else z_res2[1]
                online_max_template = preproc_torch.process(z_patch_arr2)
                max_pred_score = pred_score

            if (idx + 1) % update_interval == 0:
                # update online_template
                if online_size == 1:
                    online_template = online_max_template
                elif online_template.shape[0] < online_size:
                    online_template = torch.cat([online_template, online_max_template])
                else:
                    # replace
                    online_template[online_forget_id:online_forget_id+1] = online_max_template
                    online_forget_id = (online_forget_id + 1) % online_size
                # After update, reset aux
                max_pred_score = -1.0
                online_max_template = torch_template_input

            # Update prev_state
            prev_state = mapped_box
        return result_ok
    else:
        return result_ok


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
    parser.add_argument("--init_box_txt", type=str, default=None, help="Path to init box txt file (x0 y0 x1 y1 w h cx cy class_id)")
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image used for template/online_template")
    parser.add_argument("--search_image", type=str, default=None, help="Path to a single search image used for tracking inference")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save annotated result images")

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
            init_box_txt=args.init_box_txt,
            init_image=args.init_image,
            search_image=args.search_image,
            output_dir=args.output_dir,
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
   python export_onnx.py \
    --checkpoint checkpoints/train/mixformer2_vit/teacher_288_depth12/xxx.pth \
    --output mixformer2_vit.onnx \
    --tracker_name mixformer2_vit_online \
    --config_name 288_depth8_score

2. 测试推理并保存输入：
   python test_onnx_inference.py \
    --onnx_path mixformer2_vit.onnx \
    --checkpoint checkpoints/train/mixformer2_vit/teacher_288_depth12/xxx.pth \
    --save_inputs_dir ./test_inputs \
    --tracker_name mixformer2_vit_online \
    --config_name 288_depth8_score

3. 使用初始框、初始图片和单张搜索图片进行 ONNX 推理（并保存可视化结果）：
     python test_onnx_inference.py \
         --onnx_path mixformerv2_online_base.onnx \
         --checkpoint checkpoints/train/mixformer2_vit/teacher_288_depth12/xxx.pth \
         --tracker_name mixformer2_vit_online \
         --config_name 288_depth8_score \
         --init_box_txt ./init_box.txt \
         --init_image ./init.jpg \
         --search_image ./search_frame.jpg \
         --output_dir ./inference_results/ \
         --template_size 288 \
         --search_size 800

4. 对比误差：脚本会自动输出误差指标，如果超出阈值则警告。
"""