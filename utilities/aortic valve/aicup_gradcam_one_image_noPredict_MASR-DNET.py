# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v4.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\masrdnet_cam.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v8.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v8.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v9.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v9.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v10.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v10.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v11.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v11.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v12.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v12.png'

#MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v26.pt'
#OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\v26.png'

MODEL_PATH  = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\rt-detr.pt'
OUTPUT_FILE = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\No_Predict\rtdetr.png'

# Ảnh đơn cần visualize
IMAGE_PATH  = r'D:\OBD\AICUP25\images\test\patient0051_0284.png'
GT_PATH     = r'D:\OBD\AICUP25\labels\test\patient0051_0284.txt'

DEVICE        = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2


# -----------------------------------------------
# Wrapper để EigenCAM hoạt động với YOLO
# -----------------------------------------------
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) \
                    else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if i == len(self.model.model) - 2:
                break
        return x if not isinstance(x, (list, tuple)) else x[0]


# -----------------------------------------------
# Helper: load GT (YOLO normalized format)
# -----------------------------------------------
def load_gt(gt_path):
    if not os.path.isfile(gt_path):
        print('[WARN] GT file not found:', gt_path)
        return []
    entries = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
            entries.append((class_id, cx, cy, w, h))
    return entries


# -----------------------------------------------
# Helper: draw GT boxes (chỉ box, không tên class)
# -----------------------------------------------
def draw_gt_boxes(img_rgb, gt_entries):
    out  = img_rgb.copy()
    H, W = out.shape[:2]
    for (_, cx, cy, w, h) in gt_entries:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, BOX_THICKNESS)
    return out


# -----------------------------------------------
# 1. Load model
# -----------------------------------------------
print('Loading MASR-DNet model:', MODEL_PATH)
yolo     = YOLO(MODEL_PATH)
pt_model = yolo.model.to(DEVICE)
pt_model.eval()

wrapped = YOLOWrapper(pt_model).to(DEVICE)
wrapped.eval()

# target layer: layer -4 (C2f trước head) — giống yolo_cam
target_layers = [pt_model.model[-4]]
print('Target layer:', type(pt_model.model[-4]).__name__)
print('Model loaded.')

# -----------------------------------------------
# 2. Load image
# -----------------------------------------------
img_bgr = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, 'Cannot read image: ' + IMAGE_PATH

img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_float = np.float32(img_rgb) / 255.0
H, W      = img_rgb.shape[:2]
print('Image size: {}x{}'.format(W, H))

# -----------------------------------------------
# 3. Preprocess (YOLO: normalize về [0,1], resize 640)
# -----------------------------------------------
img_resized  = cv2.resize(img_rgb, (640, 640))
img_norm     = np.float32(img_resized) / 255.0
input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# -----------------------------------------------
# 4. EigenCAM
# -----------------------------------------------
with EigenCAM(model=wrapped, target_layers=target_layers) as cam:
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
        eigen_smooth=True,
        aug_smooth=False
    )

heatmap = cv2.resize(grayscale_cam[0], (W, H))
heatmap = np.clip(heatmap, 0, 1)
cam_rgb = show_cam_on_image(img_float, heatmap, use_rgb=True)

# -----------------------------------------------
# 5. Vẽ GT box lên heatmap
# -----------------------------------------------
gt_entries = load_gt(GT_PATH)
print('GT boxes:', len(gt_entries))

if gt_entries:
    cam_rgb = draw_gt_boxes(cam_rgb, gt_entries)

# -----------------------------------------------
# 6. Save
# -----------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
cv2.imwrite(OUTPUT_FILE, cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR))
print('Saved to:', OUTPUT_FILE)