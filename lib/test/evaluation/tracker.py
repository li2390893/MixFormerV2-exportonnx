import importlib
import os
import json
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from lib.utils.save_utils import resize_and_save, save_yolo_annotation, crop_save
from pathlib import Path


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, tracker_params=None):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(
                env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(
                env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        print(self.results_dir)
        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module(
                'lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.params = self.get_parameters(tracker_params)

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self,
                  videofilepath,
                  optional_box=None,
                  debug=None,
                  save_results=False,
                  is_zoomin=False,
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
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
            save_video: Whether to save the video with tracking results.
            yolo_model: Path to YOLOv11 model file for detection.
            yolo_conf: Confidence threshold for YOLO detection.
            yolo_classes: List of class IDs to detect (None for all classes).
            use_yolo_init: If True, use YOLO to detect initial box automatically.
            save_roi: Whether to save ROI crops centered on target.
            roi_size: Size of the ROI crop (default 64x64).
            roi_output_dir: Output directory for ROI crops.
            no_display: If True, do not display video frames (faster processing).
        """

        # params = self.get_parameters()
        params = self.params
        folder_idx = 0
        img_idx = 0
        frame_idx = 0
        roi_folder_idx = 0
        roi_img_idx = 0
        
        # Initialize YOLO detector if model is provided
        yolo_detector = None
        if yolo_model is not None:
            try:
                from ultralytics import YOLO
                yolo_detector = YOLO(yolo_model)
                print(f"YOLOv11 model loaded: {yolo_model}")
            except ImportError:
                print("Warning: ultralytics not available. YOLO detection disabled.")
                yolo_detector = None
            except Exception as e:
                print(f"Warning: Failed to load YOLO model: {e}")
                yolo_detector = None

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(
            self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(
                self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError(
                'Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(
            videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        
        # Only create window if display is enabled
        if not no_display:
            cv.namedWindow(display_name, cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 1920, 1080)
        
        success, frame = cap.read()
        # cv.imshow(display_name, frame)
        if is_zoomin:
            scale_factor = 4.0  # 放大
            h, w = frame.shape[:2]
            frame_zoom = cv.resize(
                frame, (int(w * scale_factor), int(h * scale_factor)))

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        
        # Initialize show_window early for YOLO detection
        show_window = not no_display
        
        # Variable to store detected class
        detected_class = None
        
        # Use YOLO to detect initial box if requested
        if use_yolo_init and yolo_detector is not None:
            print("Using YOLO to detect initial target...")
            detection_frame_count = 0
            max_detection_frames = 1000  # Maximum frames to search for target
            
            while detection_frame_count < max_detection_frames:
                results = yolo_detector(frame, conf=yolo_conf, classes=yolo_classes, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get the first detection (highest confidence)
                    boxes = results[0].boxes
                    box = boxes[0].xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    conf = boxes[0].conf[0].cpu().numpy()
                    cls = int(boxes[0].cls[0].cpu().numpy())
                    
                    # Store detected class for ROI folder naming
                    detected_class = cls
                    
                    # Convert to x, y, w, h format
                    x, y, x2, y2 = map(int, box)
                    w, h = x2 - x, y2 - y
                    optional_box = [x, y, w, h]
                    
                    print(f"YOLO detected object at frame {detection_frame_count}: class={cls}, conf={conf:.2f}, bbox=[{x},{y},{w},{h}]")
                    
                    # Visualize detection for user confirmation
                    if show_window:
                        frame_disp_init = frame.copy()
                        cv.rectangle(frame_disp_init, (x, y), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame_disp_init, f'Detected: class {cls}, conf {conf:.2f}', 
                                  (x, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        cv.putText(frame_disp_init, 'Press ENTER to confirm or ESC to select manually', 
                                  (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                        cv.imshow(display_name, frame_disp_init)
                        
                        key = cv.waitKey(0)
                        if key == 27:  # ESC key - manual selection
                            print("Manual selection mode...")
                            optional_box = None
                    else:
                        # Auto-confirm detection when no_display is True
                        print(f"Auto-confirmed: Detected class {cls} with confidence {conf:.2f}")
                    break  # Found target, exit detection loop
                else:
                    # No detection in current frame, try next frame
                    detection_frame_count += 1
                    if detection_frame_count % 30 == 0:  # Print progress every 30 frames
                        print(f"Searching for target... frame {detection_frame_count}")
                    
                    # Show searching status only if display is enabled
                    if show_window:
                        frame_disp_search = frame.copy()
                        cv.putText(frame_disp_search, f'Searching for target... Frame {detection_frame_count}', 
                                  (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 165, 255), 2)
                        cv.putText(frame_disp_search, 'Press ESC to select manually', 
                                  (20, 60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 165, 255), 2)
                        cv.imshow(display_name, frame_disp_search)
                        
                        key = cv.waitKey(1)
                        if key == 27:  # ESC key - manual selection
                            print("Manual selection mode...")
                            optional_box = None
                            break
                    
                    # Read next frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("Reached end of video without detecting target.")
                        optional_box = None
                        break
            
            if detection_frame_count >= max_detection_frames:
                print(f"No object detected after {max_detection_frames} frames. Please select manually.")
                optional_box = None
        
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            # raise NotImplementedError("We haven't support cv_show now.")
            while True:
                # cv.waitKey()
                # frame_disp = frame.copy()
                if is_zoomin:
                    frame_disp = frame_zoom.copy()
                else:
                    frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(
                    display_name, frame_disp, fromCenter=False)
                if is_zoomin:
                    x = round(x / scale_factor)
                    y = round(y / scale_factor)
                    w = round(w / scale_factor)
                    h = round(h / scale_factor)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        # 添加FPS计算相关变量
        start_time = time.time()
        frame_count = 0
        fps = 0

        # 如果需要，初始化视频写入器
        if save_video:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out_video = None
        else:
            out_video = None
        
        # 初始化 ROI 保存目录
        roi_base_dir = None
        roi_class_dir = None
        if save_roi:
            if roi_output_dir is None:
                # 默认在视频同级目录下创建 output 文件夹
                video_dir = os.path.dirname(os.path.abspath(videofilepath))
                roi_base_dir = os.path.join(video_dir, 'output')
            else:
                roi_base_dir = roi_output_dir
            
            if not os.path.exists(roi_base_dir):
                os.makedirs(roi_base_dir)
            
            # 如果检测到类别，创建类别文件夹
            if detected_class is not None:
                roi_class_dir = os.path.join(roi_base_dir, f'class_{detected_class}')
                if not os.path.exists(roi_class_dir):
                    os.makedirs(roi_class_dir)
                print(f"ROI images will be saved to: {roi_class_dir} (class {detected_class})")
            else:
                roi_class_dir = roi_base_dir
                print(f"ROI images will be saved to: {roi_base_dir}")
            
            video_name = Path(videofilepath).stem
        else:
            video_name = Path(videofilepath).stem

        frame_height, frame_width = frame.shape[:2]
        video_resolution = {"width": frame_width, "height": frame_height}

        results_json_data = None
        results_json_path = None
        current_results_dir = None
        roi_json_data = None
        roi_json_path = None
        current_roi_dir = None

        def finalize_sequence(json_data, json_path):
            if json_data is None or json_path is None:
                return
            if not json_data.get("tracks"):
                return
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]

            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time

            print(f"frame_idx: {frame_idx}, conf_score: {out['conf_score']}, "
                  f"FPS: {fps:.2f}")
            
            # Re-detect with YOLO if confidence is too low
            if out['conf_score'] < 0.2:
                print("Low confidence score detected.")
                if yolo_detector is not None:
                    print("Attempting to re-detect with YOLO...")
                    results = yolo_detector(frame, conf=yolo_conf, classes=yolo_classes, verbose=False)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        # Get the first detection
                        boxes = results[0].boxes
                        box = boxes[0].xyxy[0].cpu().numpy()
                        conf = boxes[0].conf[0].cpu().numpy()
                        cls = int(boxes[0].cls[0].cpu().numpy())
                        
                        # Convert to x, y, w, h format
                        x, y, x2, y2 = map(int, box)
                        w, h = x2 - x, y2 - y
                        reinit_box = [x, y, w, h]
                        
                        print(f"YOLO re-detected object: class={cls}, conf={conf:.2f}, bbox=[{x},{y},{w},{h}]")
                        print("Re-initializing tracker...")
                        
                        tracker.initialize(frame, _build_init_info(reinit_box))
                        state = reinit_box
                        output_boxes.append(state)
                    else:
                        print("YOLO could not re-detect object. Stopping tracking.")
                        break
                else:
                    print("No YOLO model available. Stopping tracking.")
                    break
            if expansion_ratio != 1.0:
                x, y, w, h = state
                # 计算新的宽度和高度
                new_w = int(w * expansion_ratio)
                new_h = int(h * expansion_ratio)
                # 调整 x, y 使得中心点不变
                x = x - (new_w - w) // 2
                y = y - (new_h - h) // 2
                # 确保不会超出图像边界
                x = max(0, x)
                y = max(0, y)
                # 更新 state
                state = [x, y, new_w, new_h]
            output_boxes.append(state)

            # 增加frame_idx计数
            frame_idx += 1

            cv.rectangle(frame_disp, (state[0], state[1]),
                         (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 2)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, f'FPS: {fps:.2f}', (20, 130),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press s to hide/show window', (20, 105),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)

            # Display the resulting frame
            if show_window:
                cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_window = not show_window
                if show_window:
                    cv.namedWindow(display_name,
                                   cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO)
                    cv.resizeWindow(display_name, 1920, 1080)
                else:
                    cv.destroyWindow(display_name)
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER',
                           (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(
                    display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

            if save_results:
                clip_dir = os.path.join(self.results_dir, 'video_{}'.format(
                    Path(videofilepath).stem))
                if not os.path.exists(clip_dir):
                    os.makedirs(clip_dir)
                video_name = Path(videofilepath).stem
                x1, y1 = state[0], state[1]
                x2, y2 = state[0] + state[2], state[1] + state[3]
                target_crop = frame[y1:y2, x1:x2]
                save_dir = os.path.join(
                    self.results_dir, f'{video_name}_{folder_idx}')
                os.makedirs(save_dir, exist_ok=True)
                if current_results_dir != save_dir:
                    current_results_dir = save_dir
                    results_json_path = os.path.join(
                        os.path.dirname(save_dir), f"{Path(save_dir).name}.json")
                    results_json_data = {
                        "resolution": video_resolution,
                        "tracks": []
                    }
                img_name = f"{img_idx}.jpg"
                if save_cls:
                    crop_save(target_crop, save_dir, img_idx)
                else:
                    resize_and_save(target_crop, save_dir, img_idx,
                                    target_size=(224, 224), keep_aspect_ratio=True)
                print(f"save {save_dir}/{img_idx}.jpg target image")
                if results_json_data is not None:
                    results_json_data["tracks"].append({
                        "image": img_name,
                        "bbox": {
                            "x": int(state[0]),
                            "y": int(state[1]),
                            "w": int(state[2]),
                            "h": int(state[3])
                        }
                    })
                img_idx += 1
                if img_idx >= max_per_folder:
                    finalize_sequence(results_json_data, results_json_path)
                    results_json_data = None
                    results_json_path = None
                    current_results_dir = None
                    folder_idx += 1
                    img_idx = 0

            if save_yolo:
                if not os.path.exists(self.results_dir):
                    os.makedirs(self.results_dir)
                # 从videofilepath中提取文件名作为保存目录的一部分
                video_name = Path(videofilepath).stem
                save_dir = os.path.join(
                    self.results_dir, f'{video_name}_yolo')
                unique_name = f"{video_name}_{frame_idx}_{int(time.time()*1000)}.jpg"
                yolo_box = [state[0], state[1], state[2], state[3], yolo_label]
                save_yolo_annotation(
                    frame, yolo_box, save_dir, unique_name
                )

            # 保存 ROI 裁剪图像
            if save_roi and roi_class_dir is not None:
                # 获取目标中心点和尺寸
                x, y, w, h = state
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 确定裁剪尺寸（正方形）
                max_side = max(w, h)
                if max_side > roi_size:
                    # 如果目标比指定尺寸大，以目标大小裁剪
                    crop_size = max_side
                else:
                    # 否则使用指定的 roi_size
                    crop_size = roi_size
                
                # 计算裁剪区域的边界
                half_crop = crop_size // 2
                x1 = center_x - half_crop
                y1 = center_y - half_crop
                x2 = x1 + crop_size
                y2 = y1 + crop_size
                
                # 获取原始帧尺寸
                frame_h, frame_w = frame.shape[:2]
                
                # 计算需要填充的区域
                pad_left = max(0, -x1)
                pad_top = max(0, -y1)
                pad_right = max(0, x2 - frame_w)
                pad_bottom = max(0, y2 - frame_h)
                
                # 调整裁剪边界到帧内
                x1_clip = max(0, x1)
                y1_clip = max(0, y1)
                x2_clip = min(frame_w, x2)
                y2_clip = min(frame_h, y2)
                
                # 裁剪帧内的部分
                roi_crop = frame[y1_clip:y2_clip, x1_clip:x2_clip].copy()
                
                # 如果需要填充
                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    roi_crop = cv.copyMakeBorder(
                        roi_crop,
                        pad_top, pad_bottom, pad_left, pad_right,
                        cv.BORDER_CONSTANT,
                        value=(128, 128, 128)
                    )
                
                # 如果裁剪尺寸不等于 roi_size，需要缩放
                if crop_size != roi_size:
                    roi_crop = cv.resize(roi_crop, (roi_size, roi_size), interpolation=cv.INTER_LINEAR)
                
                # 创建保存目录（在类别文件夹下）
                video_name = Path(videofilepath).stem
                roi_save_dir = os.path.join(roi_class_dir, f'{video_name}_{roi_folder_idx}')
                if not os.path.exists(roi_save_dir):
                    os.makedirs(roi_save_dir)
                if current_roi_dir != roi_save_dir:
                    current_roi_dir = roi_save_dir
                    roi_json_path = os.path.join(
                        os.path.dirname(roi_save_dir), f"{Path(roi_save_dir).name}.json")
                    roi_json_data = {
                        "resolution": video_resolution,
                        "tracks": []
                    }
                
                # 保存图像
                img_filename = f"{roi_img_idx:08d}.jpg"
                img_path = os.path.join(roi_save_dir, img_filename)
                cv.imwrite(img_path, roi_crop)
                if roi_json_data is not None:
                    roi_json_data["tracks"].append({
                        "image": img_filename,
                        "bbox": {
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h)
                        }
                    })
                
                # 更新计数器
                roi_img_idx += 1
                if roi_img_idx >= max_per_folder:
                    finalize_sequence(roi_json_data, roi_json_path)
                    roi_json_data = None
                    roi_json_path = None
                    current_roi_dir = None
                    roi_folder_idx += 1
                    roi_img_idx = 0

            # 如果需要，写入视频帧
            if save_video:
                if out_video is None:
                    if not os.path.exists(self.results_dir):
                        os.makedirs(self.results_dir)
                    # 初始化视频写入器
                    video_name = Path(videofilepath).stem
                    save_dir = os.path.join(
                        self.results_dir, f'{video_name}_tracked.mp4')
                    out_video = cv.VideoWriter(save_dir, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                # 写入当前帧
                out_video.write(frame_disp)

        # When everything done, release the capture
        cap.release()
        if out_video is not None:
            out_video.release()
        cv.destroyAllWindows()
        finalize_sequence(results_json_data, results_json_path)
        finalize_sequence(roi_json_data, roi_json_path)

    def get_parameters(self, tracker_params=None):
        """Get parameters."""
        param_module = importlib.import_module(
            'lib.test.parameter.{}'.format(self.name))
        search_area_scale = None
        if tracker_params is not None and 'search_area_scale' in tracker_params:
            search_area_scale = tracker_params['search_area_scale']
        model = ''
        if tracker_params is not None and 'model' in tracker_params:
            model = tracker_params['model']
        params = param_module.parameters(
            self.parameter_name, model, search_area_scale)
        if tracker_params is not None:
            for param_k, v in tracker_params.items():
                setattr(params, param_k, v)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
