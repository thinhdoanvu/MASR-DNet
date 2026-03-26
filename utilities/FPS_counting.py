import time
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
from thop import profile

#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
model = YOLO('/home/ai/yolov12/runs/detect/yolov12l_fold12/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')
#model = YOLO('/home/ai/yolov12/runs/detect/v4_aicup_train_with_test_952_681/weights/best.pt')


# ── GFLOPs & Params bằng thop ────────────────────────────
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
imgsz      = 640  # đổi nếu bạn train ở resolution khác
dummy      = torch.zeros(1, 3, imgsz, imgsz).to(device)
net        = model.model.to(device)
net.eval()

with torch.no_grad():
    macs, params = profile(net, inputs=(dummy,), verbose=False)

gflops = macs / 1e9 * 2  # MACs -> GFLOPs
print(f"Params : {params/1e6:.1f} M")
print(f"GFLOPs : {gflops:.1f}")

# ── Chuẩn bị ảnh ─────────────────────────────────────────
img_dir   = '/home/ai/AICUP25/test/images'
img_files = sorted([
    os.path.join(img_dir, f)
    for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

# ── Warm-up ───────────────────────────────────────────────
print("\nWarming up...")
for f in tqdm(img_files[:10], desc="Warm-up"):
    model.predict(f, verbose=False)

# ── Đo FPS ───────────────────────────────────────────────
start = time.time()
for f in tqdm(img_files, desc="Inference"):
    model.predict(f, verbose=False)
elapsed = time.time() - start

fps = len(img_files) / elapsed
print(f"\n" + "="*35)
print(f"  Params  : {params/1e6:.1f} M")
print(f"  GFLOPs  : {gflops:.1f}")
print(f"  FPS     : {fps:.2f}")
print(f"  Latency : {elapsed/len(img_files)*1000:.2f} ms")
print("="*35)
