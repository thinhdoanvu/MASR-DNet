import os
import torch
from ultralytics import YOLO

# ------------------------
# Config
# ------------------------
model_path = r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train\weights\best.pt"
data_dir = r"C:\Users\VU\Documents\OBD\AICUP25\test\images"
save_txt = r"C:\Users\VU\Documents\OBD\AICUP25\test\v9m_se.txt"
conf_thres = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load model
# ------------------------
model = YOLO(model_path)
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# ------------------------
# Inference + Save highest-confidence box
# ------------------------
with open(save_txt, "w") as f:
    for img_path in image_files:
        results = model(img_path, conf=conf_thres, device=device)
        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            continue

        # Chọn box có confidence cao nhất
        idx = torch.argmax(boxes.conf)
        best_box = boxes[idx]

        cls_id = int(best_box.cls.item())
        conf = float(best_box.conf.item())
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        img_name = os.path.splitext(os.path.basename(r.path))[0]
        line = f"{img_name} {cls_id} {conf:.4f} {x1} {y1} {x2} {y2}"
        print(line)
        f.write(line + "\n")

print(f"\n[✅ Done] Saved highest-confidence results to: {save_txt}")
