# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image


# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_PATH   = 'models/aicup/v4.pt'
INPUT_DIR    = 'images/aicup'
GT_DIR       = 'images/aicup/ground_truth'
OUTPUT_DIR   = 'outputs/AICUP/GRADCAM/MASR-DNet'
SCORE_THRESH = 0.3
IMG_EXTS     = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

PRED_COLOR    = (0, 255, 0)   # green
GT_COLOR      = (255, 0, 0)   # red
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICK    = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# Helper: load GT
# format: class cx cy w h  (normalized, NO image_name)
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
            cx       = float(parts[1])
            cy       = float(parts[2])
            w        = float(parts[3])
            h        = float(parts[4])
            entries.append((class_id, cx, cy, w, h))
    return entries


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
# Helper: draw GT boxes (red) - normalized cx cy w h
# -----------------------------------------------
def draw_gt_boxes(img_rgb, gt_entries, class_names=None):
    out = img_rgb.copy()
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
# Helper: save 4-panel figure
# -----------------------------------------------
def save_panel(orig_gt_rgb, orig_pred_rgb, cam_only_rgb, cam_full_rgb, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Original + GT (red)',
              'Original + Pred (green)',
              'Heatmap only',
              'Heatmap + Pred + GT']
    imgs = [orig_gt_rgb, orig_pred_rgb, cam_only_rgb, cam_full_rgb]
    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------
# 1. Load model
# -----------------------------------------------
model = YOLO(MODEL_PATH)
model.cpu()

target_layers = [model.model.model[-4]]
cam = EigenCAM(model, target_layers, task='od')

try:
    class_names = list(model.names.values())
except Exception:
    class_names = None
print('Class names:', class_names)


# -----------------------------------------------
# 2. Collect images
# -----------------------------------------------
all_images = sorted([
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(IMG_EXTS)
])
print('Found {} images in {}'.format(len(all_images), INPUT_DIR))
assert len(all_images) > 0, 'No images found!'


# -----------------------------------------------
# 3. Process all
# -----------------------------------------------
total_pred = 0
total_gt   = 0
failed     = []

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

        # EigenCAM
        grayscale_cam = cam(img_bgr)[0, :, :]
        grayscale_cam = cv2.resize(grayscale_cam, (W, H))
        grayscale_cam = np.clip(grayscale_cam, 0, 1)
        cam_only_rgb  = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # inference
        results   = model(image_path, conf=SCORE_THRESH, verbose=False)
        det       = results[0]
        boxes_np  = det.boxes.xyxy.cpu().numpy().astype(int) if det.boxes else np.zeros((0, 4), dtype=int)
        scores_np = det.boxes.conf.cpu().numpy()             if det.boxes else np.array([])
        labels_np = det.boxes.cls.cpu().numpy()              if det.boxes else np.array([])

        # GT
        gt_entries = load_gt_for_image(base, GT_DIR)
        has_gt     = len(gt_entries) > 0
        print('  [GT] {}'.format('found {} box(es)'.format(len(gt_entries)) if has_gt else 'no GT file'))

        # build images
        orig_gt_rgb   = draw_gt_boxes(img_rgb, gt_entries, class_names) if has_gt else img_rgb.copy()
        orig_pred_rgb = draw_pred_boxes(img_rgb, boxes_np, scores_np, labels_np, class_names)
        cam_full_rgb  = draw_pred_boxes(cam_only_rgb, boxes_np, scores_np, labels_np, class_names)
        if has_gt:
            cam_full_rgb = draw_gt_boxes(cam_full_rgb, gt_entries, class_names)

        # save individual
        #cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_orig_gt.png'),   cv2.cvtColor(orig_gt_rgb,   cv2.COLOR_RGB2BGR))
        #cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_orig_pred.png'), cv2.cvtColor(orig_pred_rgb, cv2.COLOR_RGB2BGR))
        #cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_heatmap.png'),   cv2.cvtColor(cam_only_rgb,  cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_heatmap_full.png'), cv2.cvtColor(cam_full_rgb, cv2.COLOR_RGB2BGR))

        # save panel
        #save_panel(orig_gt_rgb, orig_pred_rgb, cam_only_rgb, cam_full_rgb, os.path.join(OUTPUT_DIR, base + '_panel.png'))

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