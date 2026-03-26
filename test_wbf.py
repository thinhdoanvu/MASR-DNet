from ultralytics import YOLO
import os
import torch
import numpy as np
from ensemble_boxes import weighted_boxes_fusion  # pip install ensemble-boxes

# --------------------------
# Config
# --------------------------
model_paths = [
    r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train\weights\best.pt",
    r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train2\weights\best.pt",
    r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train3\weights\best.pt",
    r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train4\weights\best.pt",
    r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train5\weights\best.pt",
]
data_dir = r"C:\Users\VU\Documents\OBD\AICUP25\test\images"
save_txt = r"C:\Users\VU\Documents\OBD\AICUP25\test\yolov9m_mix_5_datasets.txt"
conf_thres = 0.25
iou_thres = 0.75  # IoU threshold cho WBF
device = 0 if torch.cuda.is_available() else "cpu"

# --------------------------
# Load models
# --------------------------
models = [YOLO(p) for p in model_paths]

# --------------------------
# Prepare list of images
# --------------------------
image_files = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]
image_files.sort()

# --------------------------
# Run WBF inference
# --------------------------
with open(save_txt, "w") as f:
    for img_path in image_files:
        boxes_list = []
        scores_list = []
        labels_list = []

        # Run inference với tất cả model
        for model in models:
            results = model(img_path, conf=conf_thres, device=device)
            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # chuẩn hóa về [0,1] theo width và height
            w, h = r.orig_shape[1], r.orig_shape[0]
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            boxes_norm = boxes_xyxy.copy()
            boxes_norm[:, 0] /= w  # x1
            boxes_norm[:, 1] /= h  # y1
            boxes_norm[:, 2] /= w  # x2
            boxes_norm[:, 3] /= h  # y2

            boxes_list.append(boxes_norm.tolist())
            scores_list.append(boxes.conf.cpu().numpy().tolist())
            labels_list.append(boxes.cls.cpu().numpy().astype(int).tolist())

        if not boxes_list:
            continue

        # Weighted Boxes Fusion
        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, iou_thr=iou_thres, skip_box_thr=conf_thres
        )

        # Chuyển lại về pixel int
        w, h = r.orig_shape[1], r.orig_shape[0]
        boxes_wbf[:, 0] = np.clip(boxes_wbf[:, 0] * w, 0, w-1)
        boxes_wbf[:, 1] = np.clip(boxes_wbf[:, 1] * h, 0, h-1)
        boxes_wbf[:, 2] = np.clip(boxes_wbf[:, 2] * w, 0, w-1)
        boxes_wbf[:, 3] = np.clip(boxes_wbf[:, 3] * h, 0, h-1)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Ghi vào file (chọn 1 box có conf cao nhất, giống trước)
        if len(boxes_wbf) > 0:
            best_idx = np.argmax(scores_wbf)
            x1, y1, x2, y2 = boxes_wbf[best_idx].astype(int)
            cls_id = int(labels_wbf[best_idx])
            conf = float(scores_wbf[best_idx])
            line = f"{img_name} {cls_id} {conf:.4f} {x1} {y1} {x2} {y2}"
            print(line)
            f.write(line + "\n")

print(f"\n[Done] Saved WBF results to {save_txt}")
