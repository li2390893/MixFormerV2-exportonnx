import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


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
              save_video=False):  # 添加save_video参数
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
        save_video: Whether to save the video with tracking results.
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
                      save_video=save_video)  # 传递save_video参数


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')

    parser.add_argument('--params__model', type=str, default=None, help="Tracking model path.")
    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument('--params__online_size', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")
    parser.add_argument('--zoomin', action='store_true', help='Whether zoom in the video.')
    parser.add_argument('--expansion_ratio', type=float, default=1.0, help="Expansion ratio for zooming in the video.")
    parser.add_argument('--max_per_folder', type=int, default=60, help="Maximum number of frames per folder for saving results.")
    parser.add_argument('--save_yolo', action='store_true', help="Whether save results in YOLO format.")
    parser.add_argument('--yolo_label', type=int, default=0, help="Label for YOLO format.")
    parser.add_argument('--save_cls', action='store_true', help="Whether save classification results (not used in video_demo.py, set to False by default).")
    parser.add_argument('--save_video', action='store_true', help="Whether to save the video with tracking results.")  # 添加save_video参数
    
    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    run_video(args.tracker_name,
              args.tracker_param,
              args.videofile,
              args.optional_box,
              args.debug,
              args.save_results,
              tracker_params=tracker_params,
              zoomin=args.zoomin,
              expansion_ratio=args.expansion_ratio,
              max_per_folder=args.max_per_folder,
              save_yolo=args.save_yolo,
              yolo_label=args.yolo_label,
              save_cls=args.save_cls,  # save_cls is not used in video_demo.py, set to False by default
              save_video=args.save_video)  # 传递save_video参数


if __name__ == '__main__':
    main()
