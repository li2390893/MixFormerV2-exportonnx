import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'out/mixformerv2_small_simplified.onnx'
RKNN_MODEL = 'out/mixformerv2_small_simplified.float.rknn'
DATASET = './track-datasets.txt'

QUANTIZE_ON = False

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0.485*255, 0.456*255, 0.406*255],[0.485*255, 0.456*255, 0.406*255],[0.485*255, 0.456*255, 0.406*255]],std_values=[[0.229*255, 0.224*255, 0.225*255],[0.229*255, 0.224*255, 0.225*255],[0.229*255, 0.224*255, 0.225*255]],quantized_algorithm='normal', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    