"""
Nhu the nay la ket qua cao nhat
"""
from ultralytics import YOLO
import os
import torch

def main():
    # ------------------------
    # Config
    # ------------------------
    model_path = r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\train2\weights\best.pt"
    data_dir = "../AICUP25/test/images/"
    save_txt = "../AICUP25/test/v4_train2.txt"
    conf_thres = 0.01
    device = 0 if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Load model
    # ------------------------
    model = YOLO(model_path)

    # ------------------------
    # Prepare list of images
    # ------------------------
    image_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    image_files.sort()

    # ------------------------
    # Run inference từng ảnh
    # ------------------------
    with open(save_txt, "w") as f:
        for img_path in image_files:
            results = model(img_path, conf=conf_thres, device=device)
            r = results[0]

            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Lấy tên file nhưng bỏ phần mở rộng
            img_name = os.path.splitext(os.path.basename(r.path))[0]

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                line = f"{img_name} {cls_id} {conf:.4f} {x1} {y1} {x2} {y2}"
                print(line)
                f.write(line + "\n")

    print(f"\n[Done] Saved results to {save_txt}")


if __name__ == "__main__":
    main()
