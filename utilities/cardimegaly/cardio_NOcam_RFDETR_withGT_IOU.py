# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rfdetr import RFDETRBase

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_WEIGHTS = '/home/ai/Cardiomegaly/rfdetr/checkpoints_cardio/checkpoint.pth'
INPUT_DIR     = 'imgs/cardiomegaly'
GT_DIR        = 'imgs/cardiomegaly/ground_truth'
OUTPUT_DIR    = 'imgs/outputs/cardiomegaly/NoCAM/RFDETR_PRED_GT'

SCORE_THRESH  = 0.3
IOU_THRESH    = 0.5
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
CLASS_NAMES   = ['Cardiomegaly']
TARGET_CLASS  = 0
OUTPUT_SIZE   = 480   # chiều rộng output tính bằng pixel, giữ tỉ lệ
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
              'Original + GT + Pred']
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
print('Loading RF-DETR model...')
rfdetr = RFDETRBase(
    num_classes=1,
    pretrain_weights=MODEL_WEIGHTS
)
print('Model loaded:', MODEL_WEIGHTS)
print('Class names  :', CLASS_NAMES)
print('Target class :', TARGET_CLASS, '=', CLASS_NAMES[TARGET_CLASS])
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
assert len(all_images) > 0, 'No images found!'

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

        # inference
        pil_img = Image.fromarray(img_rgb)
        det     = rfdetr.predict(pil_img, threshold=SCORE_THRESH)

        if det is not None and len(det) > 0:
            boxes_np  = det.xyxy.astype(int)
            scores_np = det.confidence
            labels_np = det.class_id
        else:
            boxes_np  = np.zeros((0, 4), dtype=int)
            scores_np = np.array([])
            labels_np = np.array([])

        # filter: chỉ lấy TARGET_CLASS
        if len(boxes_np) > 0:
            mask      = labels_np == TARGET_CLASS
            boxes_np  = boxes_np[mask]
            scores_np = scores_np[mask]
            labels_np = labels_np[mask]

        # chỉ lấy 1 box có score cao nhất
        if len(boxes_np) > 1:
            best_idx  = np.argmax(scores_np)
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
            iou_scores.append(iou_val)
            if matched:
                total_matched += 1
            print('  [IoU] {:.4f}  match={}'.format(iou_val, matched))

        # vẽ trên ảnh gốc trước khi resize → box sắc nét
        out_rgb = draw_pred_boxes(img_rgb, boxes_np, scores_np, labels_np, CLASS_NAMES)
        if has_gt:
            out_rgb = draw_gt_boxes(out_rgb, gt_entries, CLASS_NAMES)

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
print('Processed : {}/{}'.format(len(all_images) - len(failed), len(all_images)))
print('Total pred: {}'.format(total_pred))
print('Total GT  : {}'.format(total_gt))
print('Has both  : {}'.format(n_with_gt))
print('Matched   : {} / {} ({:.1f}%)'.format(total_matched, n_with_gt, match_rate))
print('Mean IoU  : {:.4f}'.format(mean_iou))
print('IoU thresh: {}'.format(IOU_THRESH))
print('Output dir: {}'.format(OUTPUT_DIR))
if failed:
    print('Failed ({})'.format(len(failed)))
    for f in failed:
        print('  -', f)