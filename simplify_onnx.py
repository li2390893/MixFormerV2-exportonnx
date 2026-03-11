#!/usr/bin/env python3
"""
ONNX 模型简化脚本

使用 onnxsim 库简化 ONNX 模型，去除冗余操作，优化模型结构。

依赖安装：
pip install onnx onnxsim

使用方法：
python simplify_onnx.py --input input_model.onnx --output simplified_model.onnx

参数说明：
--input: 输入 ONNX 模型路径
--output: 输出简化后 ONNX 模型路径
"""
import argparse
import onnx
from onnxsim import simplify


def simplify_onnx_model(input_path, output_path):
    """
    使用 onnxsim 简化 ONNX 模型
    
    Args:
        input_path: 输入 ONNX 模型路径
        output_path: 输出简化后 ONNX 模型路径
    """
    # 加载 ONNX 模型
    print(f"加载模型: {input_path}")
    model = onnx.load(input_path)
    
    # 简化模型
    print("正在简化模型...")
    simplified_model, check = simplify(model)
    
    # 验证简化是否成功
    assert check, "简化失败"
    print("模型简化成功")
    
    # 保存简化后的模型
    print(f"保存简化后的模型: {output_path}")
    onnx.save(simplified_model, output_path)
    print("保存完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 onnxsim 简化 ONNX 模型")
    parser.add_argument("--input", type=str, required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出简化后 ONNX 模型路径")
    
    args = parser.parse_args()
    simplify_onnx_model(args.input, args.output)
