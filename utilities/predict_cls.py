import os
import csv
from ultralytics import YOLO

# --- Config ---
model_path = "runs/classify/train2/weights/best.pt"  # đường dẫn model đã train
test_folder = "C:/Users/VU/Documents/OBD/AICUP25/test"
output_csv = "C:/Users/VU/Documents/OBD/AICUPCLS/test_predictions.csv"

# --- Load model ---
model = YOLO(model_path)

# --- Class names ---
class_names = {0: "aortic_valve", 1: "background"}  # nếu chỉ 1 class aortic_valve, thì 0=aortic_valve, 1=background hoặc ngược lại tùy cách bạn train

# --- Prepare list of test images ---
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- Predict and save ---
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'class'])

    for img_name in image_files:
        img_path = os.path.join(test_folder, img_name)
        pred = model.predict(img_path)

        # lấy top1 class index
        class_id = int(pred[0].probs.top1)
        class_label = class_names.get(class_id, "unknown")

        writer.writerow([img_name, class_label])
        print(f"{img_name} -> {class_label}")

print(f"Predictions saved to {output_csv}")
