# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_PATH   = 'models/cardiomegaly/v4.pt'
OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv4'

#MODEL_PATH   = 'models/cardiomegaly/v8.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv8'

#MODEL_PATH   = 'models/cardiomegaly/v9.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv9'

#MODEL_PATH   = 'models/cardiomegaly/v10.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv10'

#MODEL_PATH   = 'models/cardiomegaly/v11.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv11'

#MODEL_PATH   = 'models/cardiomegaly/v12.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv12'

#MODEL_PATH   = 'models/cardiomegaly/v26.pt'
#OUTPUT_DIR   = 'imgs/outputs/cardiomegaly/NoCAM_IOU/YOLOv26'

INPUT_DIR     = 'imgs/cardiomegaly'
GT_DIR        = 'imgs/cardiomegaly/ground_truth'
SCORE_THRESH  = 0.3
IOU_THRESH    = 0.5
OUTPUT_SIZE   = 480   # chiều rộng output tính bằng pixel, giữ tỉ lệ
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
CLASS_NAMES   = ['Cardiomegaly']
PRED_COLOR    = (0, 220, 0)
GT_COLOR      = (220, 0, 0)
BOX_THICKNESS = 5
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 2
FONT_THICK    = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# Helper: tính IoU giữa 2 box pixel
# -----------------------------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter   = inter_w * inter_h
    area1   = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2   = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union   = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# -----------------------------------------------
# Helper: vẽ IoU info góc trên bên trái
# -----------------------------------------------
def draw_iou_info(img_rgb, iou_val, iou_thresh):
    out   = img_rgb.copy()
    lines = [
        'IoU: {:.3f}'.format(iou_val),
        'Thr: {:.3f}'.format(iou_thresh),
    ]
    color  = (255, 255, 255)
    pad    = 10
    lh     = 60   # line height lớn để khớp với FONT_SCALE=2
    max_tw = 0
    th_val = 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
        max_tw = max(max_tw, tw)
        th_val = th
    box_w = max_tw + pad * 2
    box_h = lh * len(lines) + pad * 2
    cv2.rectangle(out, (0, 0), (box_w, box_h), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (box_w, box_h), color, 3)
    for i, ln in enumerate(lines):
        ty = pad + lh * i + th_val
        cv2.putText(out, ln, (pad, ty), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return out


# -----------------------------------------------
# Helper: resize giữ tỉ lệ
# -----------------------------------------------
def resize_keep_ratio(img, target_width):
    if target_width is None:
        return img
    h, w  = img.shape[:2]
    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# -----------------------------------------------
# Helper: load GT (YOLO normalized format)
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
        name = (class_names[int(label)]
                if class_names and int(label) < len(class_names)
                else 'cls{}'.format(int(label)))
        text = '{}: {:.0f}%'.format(name, score * 100)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), PRED_COLOR, -1)
        cv2.putText(out, text, (x1 + 1, ty), FONT, FONT_SCALE,
                    (0, 0, 0), FONT_THICK, cv2.LINE_AA)
    return out


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
        name = (class_names[class_id]
                if class_names and class_id < len(class_names)
                else 'cls{}'.format(class_id))
        text = 'GT: {}'.format(name)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        ty = min(y2 + th + 4, H - 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), GT_COLOR, -1)
        cv2.putText(out, text, (x1 + 1, ty), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return out


# -----------------------------------------------
# Helper: save 3-panel figure (optional)
# -----------------------------------------------
def save_panel(orig_gt_rgb, orig_pred_rgb, orig_both_rgb, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Original + GT (red)',
              'Original + Pred (green)',
              'Original + GT + Pred + IoU']
    imgs = [orig_gt_rgb, orig_pred_rgb, orig_both_rgb]
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
try:
    class_names = list(model.names.values())
except Exception:
    class_names = CLASS_NAMES
print('Class names  :', class_names)
print('IoU thresh   :', IOU_THRESH)
print('Output size  :', OUTPUT_SIZE, 'px wide')

# -----------------------------------------------
# 2. Collect images
# -----------------------------------------------
all_images = sorted([
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(IMG_EXTS)
])
print('Found {} images in {}'.format(len(all_images), INPUT_DIR))
assert len(all_images) > 0, 'No images found! Check INPUT_DIR: ' + INPUT_DIR

# -----------------------------------------------
# 3. Process all
# -----------------------------------------------
total_pred    = 0
total_gt      = 0
total_matched = 0
iou_scores    = []
failed        = []

for idx, image_path in enumerate(all_images):
    print('[{}/{}] {}'.format(idx + 1, len(all_images),
                              os.path.basename(image_path)))
    try:
        base    = os.path.splitext(os.path.basename(image_path))[0]
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print('  [SKIP] Cannot read:', image_path)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]

        # inference trên ảnh gốc
        results   = model(image_path, conf=SCORE_THRESH, verbose=False)
        det       = results[0]
        boxes_np  = det.boxes.xyxy.cpu().numpy().astype(int) if det.boxes else np.zeros((0, 4), dtype=int)
        scores_np = det.boxes.conf.cpu().numpy()             if det.boxes else np.array([])
        labels_np = det.boxes.cls.cpu().numpy()              if det.boxes else np.array([])

        # chỉ lấy 1 box có score cao nhất
        if len(boxes_np) > 1:
            best_idx  = int(np.argmax(scores_np))
            boxes_np  = boxes_np[best_idx:best_idx + 1]
            scores_np = scores_np[best_idx:best_idx + 1]
            labels_np = labels_np[best_idx:best_idx + 1]
        print('  [PRED] kept={}'.format(len(boxes_np)))

        # GT
        gt_entries = load_gt_for_image(base, GT_DIR)
        has_gt     = len(gt_entries) > 0
        print('  [GT] {}'.format(
            'found {} box(es)'.format(len(gt_entries)) if has_gt else 'no GT file'))

        # tính IoU
        iou_val  = 0.0
        matched  = False
        pred_box = tuple(boxes_np[0]) if len(boxes_np) > 0 else None

        if pred_box is not None and has_gt:
            best_iou = 0.0
            for (cls_id, cx, cy, bw, bh) in gt_entries:
                gx1 = int((cx - bw / 2) * W)
                gy1 = int((cy - bh / 2) * H)
                gx2 = int((cx + bw / 2) * W)
                gy2 = int((cy + bh / 2) * H)
                cur = compute_iou(pred_box, (gx1, gy1, gx2, gy2))
                if cur > best_iou:
                    best_iou = cur
            iou_val = best_iou
            matched = iou_val >= IOU_THRESH
            print('  [IoU] {:.4f}  match={}'.format(iou_val, matched))

        # vẽ trên ảnh gốc trước khi resize → box sắc nét
        out_rgb = img_rgb.copy()
        if has_gt:
            out_rgb = draw_gt_boxes(out_rgb, gt_entries, CLASS_NAMES)
        if len(boxes_np) > 0:
            out_rgb = draw_pred_boxes(out_rgb, boxes_np, scores_np, labels_np, CLASS_NAMES)
        out_rgb = draw_iou_info(out_rgb, iou_val, IOU_THRESH)

        # resize sau khi vẽ
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = resize_keep_ratio(out_bgr, OUTPUT_SIZE)

        # save
        cv2.imwrite(os.path.join(OUTPUT_DIR, base + '_pred_gt.png'), out_bgr)

        # save panel (optional)
        #orig_gt_rgb   = draw_gt_boxes(img_rgb, gt_entries, CLASS_NAMES) if has_gt else img_rgb.copy()
        #orig_pred_rgb = draw_pred_boxes(img_rgb, boxes_np, scores_np, labels_np, CLASS_NAMES)
        #save_panel(
        #    cv2.cvtColor(resize_keep_ratio(cv2.cvtColor(orig_gt_rgb,   cv2.COLOR_RGB2BGR), OUTPUT_SIZE), cv2.COLOR_BGR2RGB),
        #    cv2.cvtColor(resize_keep_ratio(cv2.cvtColor(orig_pred_rgb, cv2.COLOR_RGB2BGR), OUTPUT_SIZE), cv2.COLOR_BGR2RGB),
        #    cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB),
        #    os.path.join(OUTPUT_DIR, base + '_panel.png')
        #)

        n_pred      = len(boxes_np)
        n_gt        = len(gt_entries)
        total_pred += n_pred
        total_gt   += n_gt
        if n_gt > 0 and n_pred > 0:
            iou_scores.append(iou_val)
            if matched:
                total_matched += 1
        print('  -> pred: {}  gt: {}  iou: {:.4f}  match: {}'.format(
              n_pred, n_gt, iou_val, matched))

    except Exception as e:
        print('  [ERROR]', e)
        failed.append(image_path)

# -----------------------------------------------
# 4. Summary
# -----------------------------------------------
n_with_gt  = len(iou_scores)
mean_iou   = float(np.mean(iou_scores)) if iou_scores else 0.0
match_rate = total_matched / n_with_gt * 100 if n_with_gt > 0 else 0.0

print('\n========== DONE ==========')
print('Processed  : {}/{}'.format(len(all_images) - len(failed), len(all_images)))
print('Total pred : {}'.format(total_pred))
print('Total GT   : {}'.format(total_gt))
print('Has both   : {}'.format(n_with_gt))
print('Matched    : {} / {} ({:.1f}%)'.format(total_matched, n_with_gt, match_rate))
print('Mean IoU   : {:.4f}'.format(mean_iou))
print('IoU thresh : {}'.format(IOU_THRESH))
print('Output dir : {}'.format(OUTPUT_DIR))
if failed:
    print('Failed ({})'.format(len(failed)))
    for f in failed:
        print('  -', f)
