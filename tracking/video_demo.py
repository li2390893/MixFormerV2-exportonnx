import os
import sys
import argparse
import cv2
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker

# Try to import YOLO, but make it optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLOv11 detection will be disabled.")


def run_video(tracker_name, 
              tracker_param, 
              videofile='', 
              optional_box=None, 
              debug=None,
              save_results=False, 
              tracker_params=None, 
              zoomin=False, 
              expansion_ratio=1.0,
              max_per_folder=60, 
              save_yolo=False, 
              yolo_label=0,
              save_cls=False,
              save_video=False,
              yolo_model=None,
              yolo_conf=0.25,
              yolo_classes=None,
              use_yolo_init=False,
              save_roi=False,
              roi_size=64,
              roi_output_dir=None,
              no_display=False):
    """Run the tracker on your video with optional YOLO detection.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
        save_video: Whether to save the video with tracking results.
        yolo_model: Path to YOLOv11 model file (e.g., yolo11n.pt).
        yolo_conf: Confidence threshold for YOLO detection.
        yolo_classes: List of class IDs to detect (None for all classes).
        use_yolo_init: If True, use YOLO to detect initial box automatically.
        save_roi: Whether to save ROI crops centered on target.
        roi_size: Size of the ROI crop (default 64x64).
        roi_output_dir: Output directory for ROI crops.
        no_display: If True, do not display video frames (faster processing).
    """
    tracker = Tracker(tracker_name, tracker_param, "LASOT",
                      tracker_params=tracker_params)
    tracker.run_video(videofilepath=videofile,
                      optional_box=optional_box,
                      debug=debug,
                      save_results=save_results,
                      is_zoomin=zoomin,
                      expansion_ratio=expansion_ratio,
                      max_per_folder=max_per_folder,
                      save_yolo=save_yolo,
                      yolo_label=yolo_label,
                      save_cls=save_cls,
                      save_video=save_video,
                      yolo_model=yolo_model,
                      yolo_conf=yolo_conf,
                      yolo_classes=yolo_classes,
                      use_yolo_init=use_yolo_init,
                      save_roi=save_roi,
                      roi_size=roi_size,
                      roi_output_dir=roi_output_dir,
                      no_display=no_display)


def get_video_files(path):
    """获取视频文件列表。如果输入是文件夹，返回所有视频文件；如果是文件，返回单个文件列表。
    
    Args:
        path: 视频文件路径或文件夹路径
        
    Returns:
        视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg']
    
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        video_files = []
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        return video_files
    else:
        raise ValueError(f"路径不存在: {path}")


def main():
    parser = argparse.ArgumentParser(description='在视频上运行跟踪器（支持单个视频文件或文件夹）')
    parser.add_argument('tracker_name', type=str, help='跟踪方法名称')
    parser.add_argument('tracker_param', type=str, help='参数文件名称')
    parser.add_argument('videofile', type=str, help='视频文件路径或包含视频的文件夹路径')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='可选的初始框，格式为 x y w h')
    parser.add_argument('--debug', type=int, default=0, help='调试级别')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='保存边界框结果')

    parser.add_argument('--params__model', type=str, default=None, help="跟踪模型路径")
    parser.add_argument('--params__update_interval', type=int, default=None, help="在线跟踪的更新间隔")
    parser.add_argument('--params__online_size', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="是否可视化注意力图")
    parser.add_argument('--zoomin', action='store_true', help='是否放大视频')
    parser.add_argument('--expansion_ratio', type=float, default=1.0, help="放大视频的扩展比例")
    parser.add_argument('--max_per_folder', type=int, default=60, help="每个文件夹保存结果的最大帧数")
    parser.add_argument('--save_yolo', action='store_true', help="是否以 YOLO 格式保存结果")
    parser.add_argument('--yolo_label', type=int, default=0, help="YOLO 格式的标签")
    parser.add_argument('--save_cls', action='store_true', help="是否保存分类结果（video_demo.py 中未使用，默认为 False）")
    parser.add_argument('--save_video', action='store_true', help="是否保存带有跟踪结果的视频")
    
    # YOLOv11 检测参数
    parser.add_argument('--yolo_model', type=str, default=None, help="YOLOv11 模型文件路径（例如：yolo11n.pt）")
    parser.add_argument('--yolo_conf', type=float, default=0.25, help="YOLO 检测的置信度阈值")
    parser.add_argument('--yolo_classes', type=int, nargs='+', default=None, help="要检测的类别 ID 列表（例如：--yolo_classes 0 2 3）")
    parser.add_argument('--use_yolo_init', action='store_true', help="使用 YOLO 自动检测并初始化跟踪目标")
    
    # ROI 裁剪参数
    parser.add_argument('--save_roi', action='store_true', help="保存以目标为中心的 ROI 裁剪图像")
    parser.add_argument('--roi_size', type=int, default=64, help="ROI 裁剪的尺寸（默认 64x64）")
    parser.add_argument('--roi_output_dir', type=str, default=None, help="ROI 图像保存目录（默认为视频同级目录下的 output 文件夹）")
    
    # 显示控制参数
    parser.add_argument('--no_display', action='store_true', help="不显示视频帧（提高批量处理速度）")
    
    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) is not None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    # 获取视频文件列表
    try:
        video_files = get_video_files(args.videofile)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    if not video_files:
        print(f"未在 {args.videofile} 中找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"处理视频 [{idx}/{len(video_files)}]: {video_path}")
        print(f"{'='*60}")
        
        try:
            run_video(args.tracker_name,
                      args.tracker_param,
                      video_path,
                      args.optional_box,
                      args.debug,
                      args.save_results,
                      tracker_params=tracker_params,
                      zoomin=args.zoomin,
                      expansion_ratio=args.expansion_ratio,
                      max_per_folder=args.max_per_folder,
                      save_yolo=args.save_yolo,
                      yolo_label=args.yolo_label,
                      save_cls=args.save_cls,
                      save_video=args.save_video,
                      yolo_model=args.yolo_model,
                      yolo_conf=args.yolo_conf,
                      yolo_classes=args.yolo_classes,
                      use_yolo_init=args.use_yolo_init,
                      save_roi=args.save_roi,
                      roi_size=args.roi_size,
                      roi_output_dir=args.roi_output_dir,
                      no_display=args.no_display)
            print(f"✓ 视频处理完成: {video_path}")
        except Exception as e:
            print(f"✗ 视频处理失败: {video_path}")
            print(f"  错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"所有视频处理完成！共处理 {len(video_files)} 个视频")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
