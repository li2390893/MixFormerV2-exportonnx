import cv2
import numpy as np
import os


def save_yolo_annotation(image, box, root_dir, image_name):
    """
    保存图片和YOLO格式标注到指定目录。

    Args:
        image: OpenCV格式的BGR图片
        box: 检测框 [x, y, w, h, class_id]
        root_dir: 根目录路径
        image_name: 图片文件名（带扩展名）
    """
    # 确保根目录存在
    os.makedirs(root_dir, exist_ok=True)

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 保存图片
    img_save_path = os.path.join(images_dir, image_name)
    cv2.imwrite(img_save_path, image)

    # 计算YOLO格式 (class_id cx cy w h)，归一化到[0,1]
    h_img, w_img = image.shape[:2]
    x, y, w, h, class_id = box
    cx = (x + w / 2) / w_img
    cy = (y + h / 2) / h_img
    bw = w / w_img
    bh = h / h_img

    label_line = f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_save_path = os.path.join(labels_dir, label_name)
    with open(label_save_path, "w") as f:
        f.write(label_line + "\n")


def resize_and_save(
    target_crop,
    save_dir,
    img_idx,
    target_size=(224, 224),
    keep_aspect_ratio=True  # 新增参数，控制是否等比缩放
):
    """
    将图像缩放至目标尺寸并保存（可选择是否保持宽高比）

    参数:
        target_crop: 输入图像 (OpenCV格式, BGR)
        save_dir: 保存目录
        img_idx: 图像索引/名称
        target_size: 目标尺寸 (宽, 高), 默认为 (224, 224)
        keep_aspect_ratio: 是否保持宽高比（True=等比缩放+填充，False=直接拉伸）
    """
    # 获取原始尺寸
    h, w = target_crop.shape[:2]
    target_w, target_h = target_size

    if keep_aspect_ratio:
        # --- 等比缩放 + 填充 ---
        # 计算缩放比例并保持宽高比
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 等比缩放
        resized = cv2.resize(target_crop, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

        # 创建空白画布 (224x224)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 将缩放后的图像居中放置
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        # --- 直接拉伸缩放 ---
        canvas = cv2.resize(target_crop, (target_w, target_h),
                            interpolation=cv2.INTER_AREA)

    # 保存图像
    cv2.imwrite(os.path.join(save_dir, f"{img_idx}.jpg"), canvas)


def crop_save(
    target_crop,
    save_dir,
    img_idx,
):
    """
    将图像缩放至目标尺寸并保存（可选择是否保持宽高比）

    参数:
        target_crop: 输入图像 (OpenCV格式, BGR)
        save_dir: 保存目录
        img_idx: 图像索引/名称
    """

    # 保存图像
    cv2.imwrite(os.path.join(save_dir, f"{img_idx}.jpg"), target_crop)