#!/usr/bin/env python3
"""
Export MixFormerV2 model to ONNX format - 4 models version
Split into: Template Encoder, Online Template Encoder, Search Encoder, Tracking Head
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
import numpy as np
from onnxsim import simplify

prj_path = os.path.join(os.path.dirname(__file__), ".")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.models.mixformer2_vit import build_mixformer2_vit_online
from lib.models.mixformer2_vit.mixformer2_vit_online import VisionTransformer
from lib.models.mixformer2_vit.head import build_box_head


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
            raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")
        if not hasattr(config_module, "update_new_config_from_file"):
            raise AttributeError("Config module missing update_new_config_from_file function")
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


class TemplateBlock(nn.Module):
    """专门用于处理模板特征的Block，只进行self-attention"""
    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.drop_path2 = block.drop_path2
        self.dim = block.dim
        
    def forward(self, x):
        """x: (B, N, C) 只包含template tokens"""
        B, N, C = x.shape
        num_heads = self.attn.num_heads
        head_dim = C // num_heads
        
        qkv = self.attn.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn = attn.softmax(dim=-1)
        x_mt = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x_mt = self.attn.proj(x_mt)
        x_mt = self.attn.proj_drop(x_mt)
        
        x = x + self.drop_path1(x_mt)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class TemplateEncoder(nn.Module):
    """Template Feature Extractor - extracts features from template image"""
    def __init__(self, backbone: VisionTransformer):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.pos_embed_t = backbone.pos_embed_t
        self.pos_drop = backbone.pos_drop
        self.feat_sz_t = backbone.feat_sz_t
        self.embed_dim = backbone.patch_embed.proj.out_channels
        
        self.blocks = nn.ModuleList([TemplateBlock(blk) for blk in backbone.blocks])

    def forward(self, x_t):
        """
        x_t: (B, 3, H, W) template image
        return: (B, C, H, W) template features
        """
        x_t = self.patch_embed(x_t)
        H_t = W_t = self.feat_sz_t
        
        x_t = x_t + self.pos_embed_t
        x_t = self.pos_drop(x_t)
        
        for blk in self.blocks:
            x_t = blk(x_t)
        
        x_t = x_t.transpose(1, 2).reshape(x_t.size(0), self.embed_dim, H_t, W_t)
        return x_t


class OnlineTemplateEncoder(nn.Module):
    """Online Template Feature Extractor - extracts features from online template image"""
    def __init__(self, backbone: VisionTransformer):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.pos_embed_t = backbone.pos_embed_t
        self.pos_drop = backbone.pos_drop
        self.feat_sz_t = backbone.feat_sz_t
        self.embed_dim = backbone.patch_embed.proj.out_channels
        
        self.blocks = nn.ModuleList([TemplateBlock(blk) for blk in backbone.blocks])

    def forward(self, x_ot):
        """
        x_ot: (B, 3, H, W) online template image
        return: (B, C, H, W) online template features
        """
        x_ot = self.patch_embed(x_ot)
        H_t = W_t = self.feat_sz_t
        
        x_ot = x_ot + self.pos_embed_t
        x_ot = self.pos_drop(x_ot)
        
        for blk in self.blocks:
            x_ot = blk(x_ot)
        
        x_ot = x_ot.transpose(1, 2).reshape(x_ot.size(0), self.embed_dim, H_t, W_t)
        return x_ot


class SearchEncoder(nn.Module):
    """Search Feature Extractor - extracts features from search region with template context"""
    def __init__(self, backbone: VisionTransformer):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.pos_embed_s = backbone.pos_embed_s
        self.pos_embed_t = backbone.pos_embed_t
        self.pos_embed_reg = backbone.pos_embed_reg
        self.reg_tokens = backbone.reg_tokens
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.feat_sz_s = backbone.feat_sz_s
        self.feat_sz_t = backbone.feat_sz_t
        self.embed_dim = backbone.patch_embed.proj.out_channels

    def forward(self, x_s, x_t, x_ot, reg_tokens):
        """
        x_s: (B, 3, H_s, W_s) search image
        x_t: (B, C, H_t, W_t) template features (from TemplateEncoder)
        x_ot: (B, C, H_t, W_t) online template features (from OnlineTemplateEncoder)
        reg_tokens: (B, 4, C) regression tokens
        return: (B, C, H_s, W_s) search features, (B, 4, C) updated reg_tokens
        """
        B = x_s.size(0)
        H_s = W_s = self.feat_sz_s
        H_t = W_t = self.feat_sz_t
        
        x_s = self.patch_embed(x_s)
        x_t = x_t.flatten(2).transpose(1, 2)
        x_ot = x_ot.flatten(2).transpose(1, 2)
        
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_s = x_s + self.pos_embed_s
        reg_tokens = reg_tokens + self.pos_embed_reg
        
        x = torch.cat([x_t, x_ot, x_s, reg_tokens], dim=1)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, H_t, W_t, H_s, W_s)
        
        x_t_out = x[:, :H_t*W_t]
        x_ot_out = x[:, H_t*W_t:H_t*W_t*2]
        x_s_out = x[:, H_t*W_t*2:H_t*W_t*2 + H_s*W_s]
        reg_tokens_out = x[:, H_t*W_t*2 + H_s*W_s:]
        
        x_s_2d = x_s_out.transpose(1, 2).reshape(B, self.embed_dim, H_s, W_s)
        return x_s_2d, reg_tokens_out


class TrackingHead(nn.Module):
    """Tracking Head - predicts box and score from search features"""
    def __init__(self, cfg, model=None):
        super().__init__()
        if model is not None:
            self.box_head = model.box_head.cpu()
            self.score_head = model.score_head.cpu()
            self.head_type = model.head_type
            self.embed_dim = model.backbone.patch_embed.proj.out_channels
        else:
            self.head_type = cfg.MODEL.HEAD_TYPE
            hidden_dim = cfg.MODEL.HIDDEN_DIM
            self.box_head = build_box_head(cfg).cpu()
            self.score_head = None
            self.embed_dim = hidden_dim
        
    def forward(self, search_feat, reg_tokens, softmax=True):
        """
        search_feat: (B, C, H_s, W_s) search features from SearchEncoder
        reg_tokens: (B, 4, C) regression tokens (updated by SearchEncoder)
        return: pred_boxes, pred_scores
        """
        B = search_feat.size(0)
        
        pred_boxes, prob_l, prob_t, prob_r, prob_b = self.box_head(reg_tokens, softmax=softmax)
        
        from lib.utils.box_ops import box_xyxy_to_cxcywh
        outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
        outputs_coord_new = outputs_coord.view(B, 1, 4)
        
        if self.score_head is not None:
            scores = self.score_head(reg_tokens).view(-1)
            scores = torch.sigmoid(scores)
        else:
            scores = torch.ones(B)
        
        return outputs_coord_new, scores


def export_template_encoder(cfg, checkpoint_path, template_size, output_path, opset_version=17):
    """Export Template Encoder - stores qkv_mem for later use"""
    print(f"\n{'='*50}")
    print("Exporting Template Encoder...")
    print(f"{'='*50}")
    
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    model.cpu()
    
    template_encoder = TemplateEncoder(model.backbone)
    template_encoder.eval()
    template_encoder.cpu()
    
    H, W = _normalize_hw(template_size)
    template_input = torch.randn(1, 3, H, W)
    
    torch.onnx.export(
        template_encoder,
        template_input,
        output_path,
        input_names=["template"],
        output_names=["template_feat"],
        opset_version=opset_version,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )
    
    print(f"Template Encoder saved to: {output_path}")
    
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {"template": template_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"Template Encoder output shape: {ort_outputs[0].shape}")
    
    return template_encoder


def export_online_template_encoder(cfg, checkpoint_path, template_size, output_path, opset_version=17):
    """Export Online Template Encoder"""
    print(f"\n{'='*50}")
    print("Exporting Online Template Encoder...")
    print(f"{'='*50}")
    
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    model.cpu()
    
    online_template_encoder = OnlineTemplateEncoder(model.backbone)
    online_template_encoder.eval()
    online_template_encoder.cpu()
    
    H, W = _normalize_hw(template_size)
    online_template_input = torch.randn(1, 3, H, W)
    
    torch.onnx.export(
        online_template_encoder,
        online_template_input,
        output_path,
        input_names=["online_template"],
        output_names=["online_template_feat"],
        opset_version=opset_version,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )
    
    print(f"Online Template Encoder saved to: {output_path}")
    
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {"online_template": online_template_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"Online Template Encoder output shape: {ort_outputs[0].shape}")
    
    return True


def export_search_encoder(cfg, checkpoint_path, template_size, search_size, output_path, opset_version=17):
    """Export Search Encoder"""
    print(f"\n{'='*50}")
    print("Exporting Search Encoder...")
    print(f"{'='*50}")
    
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    model.cpu()
    
    search_encoder = SearchEncoder(model.backbone)
    search_encoder.eval()
    search_encoder.cpu()
    
    H_t = model.backbone.feat_sz_t
    W_t = model.backbone.feat_sz_t
    H_s = model.backbone.feat_sz_s
    W_s = model.backbone.feat_sz_s
    embed_dim = model.backbone.patch_embed.proj.out_channels
    
    search_input = torch.randn(1, 3, H_s * 16, W_s * 16)
    template_feat = torch.randn(1, embed_dim, H_t, W_t)
    online_template_feat = torch.randn(1, embed_dim, H_t, W_t)
    reg_tokens = torch.randn(1, 4, embed_dim)
    
    torch.onnx.export(
        search_encoder,
        (search_input, template_feat, online_template_feat, reg_tokens),
        output_path,
        input_names=["search", "template_feat", "online_template_feat", "reg_tokens"],
        output_names=["search_feat", "reg_tokens_out"],
        opset_version=opset_version,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )
    
    print(f"Search Encoder saved to: {output_path}")
    
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {
        "search": search_input.numpy(),
        "template_feat": template_feat.numpy(),
        "online_template_feat": online_template_feat.numpy(),
        "reg_tokens": reg_tokens.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"Search Encoder output shapes: search_feat={ort_outputs[0].shape}, reg_tokens={ort_outputs[1].shape}")
    
    return True


def export_tracking_head(cfg, checkpoint_path, search_size, output_path, opset_version=17):
    """Export Tracking Head"""
    print(f"\n{'='*50}")
    print("Exporting Tracking Head...")
    print(f"{'='*50}")
    
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    model.cpu()
    
    tracking_head = TrackingHead(cfg, model)
    tracking_head.eval()
    tracking_head.cpu()
    
    H_s = model.backbone.feat_sz_s
    W_s = model.backbone.feat_sz_s
    embed_dim = model.backbone.patch_embed.proj.out_channels
    
    search_feat = torch.randn(1, embed_dim, H_s, W_s)
    reg_tokens = torch.randn(1, 4, embed_dim)
    
    torch.onnx.export(
        tracking_head,
        (search_feat, reg_tokens),
        output_path,
        input_names=["search_feat", "reg_tokens"],
        output_names=["pred_boxes", "pred_scores"],
        opset_version=opset_version,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )
    
    print(f"Tracking Head saved to: {output_path}")
    
    return True


def export_all_models(
    cfg,
    checkpoint_path,
    output_dir,
    template_size,
    search_size,
    online_template_size=None,
    opset_version=17,
):
    """Export all 4 models"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    template_encoder_path = os.path.join(output_dir, "template_encoder.onnx")
    online_template_encoder_path = os.path.join(output_dir, "online_template_encoder.onnx")
    search_encoder_path = os.path.join(output_dir, "search_encoder.onnx")
    tracking_head_path = os.path.join(output_dir, "tracking_head.onnx")
    
    model = build_mixformer2_vit_online(cfg, train=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()
    
    embed_dim = model.backbone.patch_embed.proj.out_channels
    reg_tokens = model.backbone.reg_tokens.data.numpy()
    
    reg_tokens_path = os.path.join(output_dir, "reg_tokens.npy")
    np.save(reg_tokens_path, reg_tokens)
    print(f"reg_tokens saved to: {reg_tokens_path}")
    print(f"reg_tokens shape: {reg_tokens.shape}")
    
    export_template_encoder(cfg, checkpoint_path, template_size, template_encoder_path, opset_version)
    export_online_template_encoder(cfg, checkpoint_path, template_size, online_template_encoder_path, opset_version)
    export_search_encoder(cfg, checkpoint_path, template_size, search_size, search_encoder_path, opset_version)
    export_tracking_head(cfg, checkpoint_path, search_size, tracking_head_path, opset_version)
    
    print(f"\n{'='*50}")
    print("All models exported successfully!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"  - template_encoder.onnx")
    print(f"  - online_template_encoder.onnx")
    print(f"  - search_encoder.onnx")
    print(f"  - tracking_head.onnx")
    print(f"  - reg_tokens.npy")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Export MixFormerV2 4-models to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="out_4", help="Output directory")
    parser.add_argument("--template_size", type=int, default=112, help="Template size")
    parser.add_argument("--search_size", type=int, default=224, help="Search size")
    parser.add_argument("--online_template_size", type=int, default=None, help="Online template size")
    parser.add_argument("--tracker_name", type=str, default="mixformer2_vit_online", help="Tracker name")
    parser.add_argument("--config_name", type=str, default="base_config", help="Config name")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    
    args = parser.parse_args()
    
    cfg = load_config(
        args.tracker_name,
        args.config_name,
        args.config_path,
        prj_path,
    )
    
    template_size = args.template_size
    search_size = args.search_size
    online_template_size = args.online_template_size or template_size
    
    export_all_models(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        template_size=template_size,
        search_size=search_size,
        online_template_size=online_template_size,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    print("想要正确导出需要修改lib/models/mixformer2_vit/head.py的 原始的模型是在cuda上运行的，这里将indice也放到cpu上"
            "self.indice = torch.arange(0, feat_sz).unsqueeze(0).cuda() * stride # (1, feat_sz)"
            "# self.indice = torch.arange(0, feat_sz).unsqueeze(0) * stride # (1, feat_sz)"
        )
    main()

"""
python export_onnx_4.py \
    --checkpoint ./models/mixformerv2_small.pth.tar \
    --output_dir out_4 \
    --template_size 112 \
    --search_size 224 \
    --tracker_name mixformer2_vit_online \
    --config_name 224_depth4_mlp1_score \
    --opset_version 11
"""