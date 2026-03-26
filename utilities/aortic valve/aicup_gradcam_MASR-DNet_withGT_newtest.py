# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_PATH   = r'D:\OBD\Ultility\YOLO-V8-CAM\models\aicup\v4.pt'
INPUT_DIR    = r'D:\OBD\AICUP25\test\newtest'
GT_DIR       = r'D:\OBD\AICUP25\test\lbl\patient0100'
OUTPUT_DIR   = r'D:\OBD\Ultility\YOLO-V8-CAM\outputs\AICUP\GRADCAM\MASR-DNet_newtest'

SCORE_THRESH  = 0.3
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
DEVICE        = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GT_COLOR      = (255, 0, 0)
PRED_COLOR    = (0, 255, 0)
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICK    = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# Wrapper YOLO (handle skip connections)
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
# Helper: load GT
# -----------------------------------------------
def load_gt_for_image(base, gt_dir):
    gt_path = os.path.join(gt_dir, base + '.txt')
    if not os.path.isfile(gt_path):
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
# Helper: draw GT boxes (red)
# -----------------------------------------------
def draw_gt_boxes(img_rgb, gt_entries, class_names=None):
    out  = img_rgb.copy()
    H, W = out.shape[:2]
    for (class_id, cx, cy, w, h) in gt_entries:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, BOX_THICKNESS)
        name = class_names[class_id] if class_names and class_id < len(class_names) else 'cls{}'.format(class_id)
        text = 'GT: {}'.format(name)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        ty = min(y2 + th + 4, H - 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), GT_COLOR, -1)
        cv2.putText(out, text, (x1 + 1, ty), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return out


# -----------------------------------------------
# Helper: draw prediction boxes (green)
# -----------------------------------------------
def draw_pred_boxes(img_rgb, boxes, scores, labels, class_names=None):
    out = img_rgb.copy()
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), PRED_COLOR, BOX_THICKNESS)
        name = class_names[int(label)] if class_names and int(label) < len(class_names) else 'cls{}'.format(int(label))
        text = '{}: {:.0f}%'.format(name, score * 100)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), PRED_COLOR, -1)
        cv2.putText(out, text, (x1 + 1, ty), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)
    return out


# -----------------------------------------------
# 1. Load model
# -----------------------------------------------
print('Loading model:', MODEL_PATH)
yolo     = YOLO(MODEL_PATH)
pt_model = yolo.model.to(DEVICE)
pt_model.eval()

wrapped       = YOLOWrapper(pt_model).to(DEVICE)
wrapped.eval()
target_layers = [pt_model.model[-4]]

try:
    class_names = list(yolo.names.values())
except Exception:
    class_names = None
print('Class names  :', class_names)
print('Target layer :', type(pt_model.model[-4]).__name__)
print('Device       :', DEVICE)


# -----------------------------------------------
# 2. Collect images
# -----------------------------------------------
all_images = sorted([
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(IMG_EXTS)
])
print('Found {} images in {}'.format(len(all_images), INPUT_DIR))
assert len(all_images) > 0, 'No images found! Check: ' + INPUT_DIR


# -----------------------------------------------
# 3. Process all
# -----------------------------------------------
total_pred = 0
total_gt   = 0
failed     = []

with EigenCAM(model=wrapped, target_layers=target_layers) as cam:
    for idx, image_path in enumerate(all_images):
        print('[{}/{}] {}'.format(idx + 1, len(all_images), os.path.basename(image_path)))
        try:
            base = os.path.splitext(os.path.basename(image_path))[0]

            # load
            img_bgr   = cv2.imread(image_path)
            if img_bgr is None:
                print('  [SKIP] Cannot read:', image_path)
                continue
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = np.float32(img_rgb) / 255.0
            H, W      = img_rgb.shape[:2]

            # preprocess: resize 640, normalize [0,1]
            img_resized  = cv2.resize(img_rgb, (640, 640))
            img_norm     = np.float32(img_resized) / 255.0
            input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # EigenCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                                eigen_smooth=True, aug_smooth=False)
            heatmap      = cv2.resize(grayscale_cam[0].astype(np.float32), (W, H))
            heatmap      = np.clip(heatmap, 0, 1)
            cam_only_rgb = show_cam_on_image(img_float, heatmap, use_rgb=True)

            # inference
            results   = yolo(image_path, conf=SCORE_THRESH, verbose=False)
            det       = results[0]
            boxes_np  = det.boxes.xyxy.cpu().numpy().astype(int) if det.boxes else np.zeros((0, 4), dtype=int)
            scores_np = det.boxes.conf.cpu().numpy()             if det.boxes else np.array([])
            labels_np = det.boxes.cls.cpu().numpy()              if det.boxes else np.array([])

            print('  [PRED] {}'.format(len(boxes_np)))

            # GT
            gt_entries = load_gt_for_image(base, GT_DIR)
            has_gt     = len(gt_entries) > 0
            print('  [GT] {}'.format('found {} box(es)'.format(len(gt_entries)) if has_gt else 'no GT file'))

            # build output image
            cam_full_rgb = draw_pred_boxes(cam_only_rgb, boxes_np, scores_np, labels_np, class_names)
            if has_gt:
                cam_full_rgb = draw_gt_boxes(cam_full_rgb, gt_entries, class_names)

            # save
            cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_heatmap_full.png'),
                        cv2.cvtColor(cam_full_rgb, cv2.COLOR_RGB2BGR))

            n_pred = len(boxes_np)
            n_gt   = len(gt_entries)
            total_pred += n_pred
            total_gt   += n_gt
            print('  -> pred: {}  gt: {}'.format(n_pred, n_gt))

        except Exception as e:
            print('  [ERROR]', e)
            failed.append(image_path)


# -----------------------------------------------
# 4. Summary
# -----------------------------------------------
print('\n========== DONE ==========')
print('Processed : {}/{}'.format(len(all_images) - len(failed), len(all_images)))
print('Total pred: {}'.format(total_pred))
print('Total GT  : {}'.format(total_gt))
print('Output dir: {}'.format(OUTPUT_DIR))
if failed:
    print('Failed ({})'.format(len(failed)))
    for f in failed:
        print('  -', f)