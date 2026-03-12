import os
import sys
import argparse
import numpy as np
import cv2 as cv
import onnxruntime

prj_path = os.path.join(os.path.dirname(__file__), ".")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.tracker.tracker_utils import PreprocessorX_onnx
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


drawing = False
ix, iy = -1, -1
init_box = None
selection_rect = None
tracking_started = False
first_frame_paused = True


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, init_box, selection_rect
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        selection_rect = None
        
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            selection_rect = (ix, iy, x, y)
            
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = ix, iy
        x2, y2 = x, y
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w > 10 and h > 10:
            init_box = [float(x_min), float(y_min), float(w), float(h)]
        selection_rect = None


def draw_box(image, box, score=None, color=(0, 255, 0)):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
    if score is not None:
        cv.putText(image, f"{score:.3f}", (int(x1), int(y1)-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def denormalize_box(box, img_h, img_w):
    cx, cy, w, h = box
    cx = cx * img_w
    cy = cy * img_h
    w = w * img_w
    h = h * img_h
    x = cx - w / 2
    y = cy - h / 2
    return [x, y, w, h]


def map_box_back(pred_box, prev_state, resize_factor, search_sz):
    cx_prev = prev_state[0] + 0.5 * prev_state[2]
    cy_prev = prev_state[1] + 0.5 * prev_state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_sz / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def main():
    parser = argparse.ArgumentParser(description="ONNX tracking with interactive box selection")
    parser.add_argument("--onnx_path", type=str, default="out/mixformerv2_small_simplified.onnx", help="Path to ONNX model")
    parser.add_argument("--video", type=str, default="videos/track-car4.mp4", help="Path to video file or camera index")
    parser.add_argument("--template_size", type=int, default=112, help="Template size")
    parser.add_argument("--search_size", type=int, default=224, help="Search size")
    parser.add_argument("--template_factor", type=float, default=2.0, help="Template factor")
    parser.add_argument("--search_factor", type=float, default=4.5, help="Search factor")
    parser.add_argument("--save_dir", type=str, default="debug_inputs", help="Directory to save input images")
    args = parser.parse_args()

    template_size = args.template_size
    search_size = args.search_size
    template_factor = args.template_factor
    search_factor = args.search_factor

    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    preproc_x = PreprocessorX_onnx()

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        exit()

    global init_box, selection_rect, tracking_started, first_frame_paused
    
    template_input = None
    online_template_input = None
    prev_state = None
    quit_flag = False
    
    import time
    fps_start_time = None
    frame_count = 0
    current_fps = 0
    total_frame_count = 0
    
    print("Instructions:")
    print("  - Draw a box with mouse to select target on first frame")
    print("  - Tracking will start automatically after drawing box")
    print("  - Press 'r' to reset selection")
    print("  - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        h, w = frame.shape[:2]

        if selection_rect is not None:
            x1, y1, x2, y2 = selection_rect
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if not tracking_started and init_box is not None:
            draw_box(display_frame, init_box, color=(0, 255, 0))

        if tracking_started and template_input is not None:
            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            x_res = sample_target(img_rgb, prev_state, search_factor, output_sz=search_size)
            x_patch_arr = x_res[0]
            resize_factor = x_res[1]
            x_amask_arr = x_res[2] if len(x_res) > 2 else x_res[1]
            img_search_input, _ = preproc_x.process(x_patch_arr, np.asarray(x_amask_arr))

            ort_inputs = {
                "template": template_input.astype(np.float32),
                "online_template": online_template_input.astype(np.float32),
                "search": img_search_input.astype(np.float32),
            }
            
            ort_outs = ort_session.run(None, ort_inputs)
            pred_boxes, pred_scores = ort_outs
            
            if pred_boxes.ndim == 3:
                mean_box = pred_boxes.mean(axis=1).reshape(-1)
            elif pred_boxes.ndim == 2 and pred_boxes.shape[1] % 4 == 0:
                N = pred_boxes.shape[1] // 4
                boxes_reshaped = pred_boxes.reshape(1, N, 4)
                mean_box = boxes_reshaped.mean(axis=1).reshape(-1)
            else:
                mean_box = pred_boxes[0]
            
            pred_box = mean_box * search_size / resize_factor
            score = pred_scores[0, 0] if pred_scores.ndim > 1 else pred_scores[0]
            
            pred_box_map = map_box_back(pred_box.tolist(), prev_state, resize_factor, search_size)
            pred_box_map = clip_box(pred_box_map, h, w, margin=10)
            
            prev_state = pred_box_map
            
            draw_box(display_frame, pred_box_map, score)
            
            # 更新模板
            # 每200帧更新一次模板
            total_frame_count += 1
            if total_frame_count % 200 == 0:
                if(score > 0.5):
                    z_res = sample_target(img_rgb, prev_state, template_factor, output_sz=template_size)
                    z_patch_arr = z_res[0]
                    z_amask_arr = z_res[2] if len(z_res) > 2 else z_res[1]
                    online_template_input, _ = preproc_x.process(z_patch_arr, np.asarray(z_amask_arr))
            
            frame_count += 1
            if fps_start_time is None:
                fps_start_time = time.time()
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
        
        cv.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv.imshow('Tracking', display_frame)
        cv.setMouseCallback('Tracking', mouse_callback)

        # 等待用户确认框选
        if first_frame_paused:
            while first_frame_paused:
                temp_frame = frame.copy()
                
                if selection_rect is not None:
                    x1, y1, x2, y2 = selection_rect
                    cv.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                if init_box is not None:
                    draw_box(temp_frame, init_box, color=(0, 255, 0))
                
                cv.imshow('Tracking', temp_frame)
                cv.setMouseCallback('Tracking', mouse_callback)
                
                key = cv.waitKey(10) & 0xFF
                
                if key == ord('q'):
                    quit_flag = True
                    break
                if key == ord('r'):
                    init_box = None
                    print("Selection reset. Draw a new box.")
                
                if init_box is not None:
                    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    
                    save_dir = args.save_dir
                    os.makedirs(save_dir, exist_ok=True)
                    
                    z_res = sample_target(img_rgb, init_box, template_factor, output_sz=template_size)
                    z_patch_arr = z_res[0]
                    z_amask_arr = z_res[2] if len(z_res) > 2 else z_res[1]
                    template_input, _ = preproc_x.process(z_patch_arr, np.asarray(z_amask_arr))
                    online_template_input = template_input.copy()
                    prev_state = init_box
                    
                    cv.imwrite(os.path.join(save_dir, "template_patch.png"), z_patch_arr)
                    
                    x_res = sample_target(img_rgb, prev_state, search_factor, output_sz=search_size)
                    x_patch_arr = x_res[0]
                    resize_factor = x_res[1]
                    x_amask_arr = x_res[2] if len(x_res) > 2 else x_res[1]
                    img_search_input, _ = preproc_x.process(x_patch_arr, np.asarray(x_amask_arr))
                    
                    cv.imwrite(os.path.join(save_dir, "search_patch.png"), x_patch_arr)
                    print(f"Input images saved to {save_dir}")

                    ort_inputs = {
                        "template": template_input.astype(np.float32),
                        "online_template": online_template_input.astype(np.float32),
                        "search": img_search_input.astype(np.float32),
                    }
                    
                    ort_outs = ort_session.run(None, ort_inputs)
                    pred_boxes, pred_scores = ort_outs
                    
                    if pred_boxes.ndim == 3:
                        mean_box = pred_boxes.mean(axis=1).reshape(-1)
                    elif pred_boxes.ndim == 2 and pred_boxes.shape[1] % 4 == 0:
                        N = pred_boxes.shape[1] // 4
                        boxes_reshaped = pred_boxes.reshape(1, N, 4)
                        mean_box = boxes_reshaped.mean(axis=1).reshape(-1)
                    else:
                        mean_box = pred_boxes[0]
                    
                    pred_box = mean_box * search_size / resize_factor
                    score = pred_scores[0, 0] if pred_scores.ndim > 1 else pred_scores[0]
                    
                    pred_box_map = map_box_back(pred_box.tolist(), prev_state, resize_factor, search_size)
                    pred_box_map = clip_box(pred_box_map, h, w, margin=10)
                    
                    prev_state = pred_box_map
                    
                    draw_box(display_frame, pred_box_map, score)
                    
                    tracking_started = True
                    first_frame_paused = False
                    print(f"Tracking started! Initial box: {init_box}")
                    break
        
        if quit_flag:
            break
            
        if not first_frame_paused:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                init_box = None
                template_input = None
                online_template_input = None
                prev_state = None
                tracking_started = False
                first_frame_paused = True
                print("Selection reset. Draw a new box on next frame.")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
