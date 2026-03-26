import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

# -----------------------------
# Đường dẫn
# -----------------------------
IMAGE_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\gin_train\images"
LABEL_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\gin_train\labels"
OUT_IMG = r"C:\Users\VU\Documents\OBD\AICUP25\gin_train\aug_4crop_bright"
OUT_LABEL = r"C:\Users\VU\Documents\OBD\AICUP25\gin_train\aug_4crop_bright_labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

# -----------------------------
# Image size
# -----------------------------
img_h, img_w = 512, 512

# -----------------------------
# Augmentation pipeline (Albumentations mới)
# -----------------------------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),

    # Affine shift/scale/rotate
    A.Affine(
        translate_percent={"x": (-0.15,0.15), "y": (-0.15,0.15)},
        scale=(0.85, 1.15),
        rotate=(-15, 15),
        mode=cv2.BORDER_CONSTANT,
        p=0.7
    ),

    # Brightness / Contrast
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

    # Hue/Saturation/Value
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),

    # Gaussian noise
    # A.GaussNoise(sigma=(5.0, 20.0), p=0.3),

    # RandomResizedCrop nhẹ, giữ object
    A.RandomResizedCrop(size=(img_h,img_w), scale=(0.8,1.0), ratio=(0.9,1.1), p=0.5),

    # Resize cuối cùng để chắc chắn
    A.Resize(height=img_h, width=img_w)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -----------------------------
# Hàm đọc label YOLO -> pascal_voc
# -----------------------------
def read_yolo_label(label_path, img_w, img_h):
    bboxes = []
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x_c, y_c, w, h = map(float, parts)
            x1 = (x_c - w/2) * img_w
            y1 = (y_c - h/2) * img_h
            x2 = (x_c + w/2) * img_w
            y2 = (y_c + h/2) * img_h
            bboxes.append([x1, y1, x2, y2])
            labels.append(int(cls))
    return bboxes, labels

# -----------------------------
# Hàm lưu label pascal_voc -> YOLO
# -----------------------------
def save_yolo_label(label_path, bboxes, labels, img_w, img_h):
    lines = []
    for bbox, cls in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        x_c = ((x1 + x2)/2)/img_w
        y_c = ((y1 + y2)/2)/img_h
        w = (x2 - x1)/img_w
        h = (y2 - y1)/img_h
        lines.append(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    with open(label_path, 'w') as f:
        f.writelines(lines)

# -----------------------------
# Thực hiện augment (chỉ ảnh có label)
# -----------------------------
for img_file in os.listdir(IMAGE_DIR):
    if not img_file.lower().endswith(('.jpg','.png','.jpeg')):
        continue

    label_path = os.path.join(LABEL_DIR, Path(img_file).stem + ".txt")
    if not os.path.exists(label_path):
        # Không có label -> bỏ qua
        continue

    # Đọc label để kiểm tra có bbox không
    bboxes, labels = read_yolo_label(label_path, img_w, img_h)
    if len(bboxes) == 0:
        continue  # label trống -> bỏ qua

    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Augment 4 lần
    for i in range(4):
        transformed = transform(image=img, bboxes=bboxes, labels=labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['labels']

        # Tạo tên file mới tránh trùng
        base_name = Path(img_file).stem
        out_img_name = f"{base_name}_aug{i+1}.jpg"
        out_label_name = f"{base_name}_aug{i+1}.txt"

        out_img_path = os.path.join(OUT_IMG, out_img_name)
        out_label_path = os.path.join(OUT_LABEL, out_label_name)

        cv2.imwrite(out_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        save_yolo_label(out_label_path, aug_bboxes, aug_labels, img_w, img_h)

print("✅ Done augmentation tất cả ảnh có label!")

