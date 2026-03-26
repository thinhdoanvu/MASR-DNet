# encoding: utf-8
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector

# -----------------------------------------------
# CONFIG 
# -----------------------------------------------
#CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/detr-1/config_aicup25_detr.py'
#CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/detr-1/epoch_3.pth'
#OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/NoCAM_IOU/DETR_PRED_GT'

#CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/dino-1/config_aicup25_dino.py'
#CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/dino-1/best_coco_bbox_mAP_epoch_2.pth'
#OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/NoCAM_IOU/DINO_PRED_GT'

#CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/swinB-1/config_aicup25_swinB.py'
#CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/swinB-1/epoch_4.pth'
#OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/NoCAM_IOU/SWINB_PRED_GT'

INPUT_DIR     = '/home/ai/mmdetection3x/imgs/cardiomegaly'
GT_DIR        = '/home/ai/mmdetection3x/imgs/cardiomegaly/ground_truth'
DEVICE        = 'cuda:0'
SCORE_THRESH  = 0.3
IOU_THRESH    = 0.5
TARGET_SIZE   = 480   # resize chiều ngắn nhất về giá trị này
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
CLASS_NAMES   = ['Cardiomegaly']
PRED_COLOR    = (0, 255, 0)
GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICK    = 1
TMP_PATH      = '/tmp/mmdet_resize_tmp.jpg'  # file tạm cho inference


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
    pad    = 6
    lh     = 20
    max_tw = 0
    th_val = 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
        max_tw = max(max_tw, tw)
        th_val = th
    box_w = max_tw + pad * 2
    box_h = lh * len(lines) + pad * 2
    cv2.rectangle(out, (0, 0), (box_w, box_h), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (box_w, box_h), color, 2)
    for i, ln in enumerate(lines):
        ty = pad + lh * i + th_val
        cv2.putText(out, ln, (pad, ty), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return out


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
            cx = float(parts[1])
            cy = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
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
# Helper: resize giữ tỉ lệ, chiều ngắn = TARGET_SIZE
# -----------------------------------------------
def resize_keep_ratio(img_rgb, target=TARGET_SIZE):
    H, W = img_rgb.shape[:2]
    if min(H, W) == target:
        return img_rgb
    scale = target / min(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)
    return cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)


# -----------------------------------------------
# Helper: process single image
# -----------------------------------------------
def process_image(image_path, full_model, class_names, gt_dir, output_dir):
    base    = os.path.splitext(os.path.basename(image_path))[0]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print('  [SKIP] Cannot read:', image_path)
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Resize về TARGET_SIZE px chiều ngắn nhất ─────
    img_rgb = resize_keep_ratio(img_rgb, TARGET_SIZE)
    H, W    = img_rgb.shape[:2]

    # Lưu ảnh đã resize ra file tạm để inference_detector đọc đúng kích thước
    cv2.imwrite(TMP_PATH, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    # ─────────────────────────────────────────────────

    # inference trên ảnh đã resize
    det_result = inference_detector(full_model, TMP_PATH)
    pred       = det_result.pred_instances
    keep       = pred.scores > SCORE_THRESH
    boxes_np   = pred.bboxes[keep].cpu().numpy().astype(int)
    scores_np  = pred.scores[keep].cpu().numpy()
    labels_np  = pred.labels[keep].cpu().numpy()

    # chỉ lấy 1 box có score cao nhất
    if len(boxes_np) > 1:
        best_idx  = np.argmax(scores_np)
        boxes_np  = boxes_np[best_idx:best_idx + 1]
        scores_np = scores_np[best_idx:best_idx + 1]
        labels_np = labels_np[best_idx:best_idx + 1]
    print('  [PRED] kept={}'.format(len(boxes_np)))

    # GT
    gt_entries = load_gt_for_image(base, gt_dir)
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

    # build output image
    out_rgb = img_rgb.copy()
    if has_gt:
        out_rgb = draw_gt_boxes(out_rgb, gt_entries, class_names)
    if len(boxes_np) > 0:
        out_rgb = draw_pred_boxes(out_rgb, boxes_np, scores_np, labels_np, class_names)
    out_rgb = draw_iou_info(out_rgb, iou_val, IOU_THRESH)

    # save
    save_path = os.path.join(output_dir, base + '_pred_gt.png')
    cv2.imwrite(save_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    return len(boxes_np), len(gt_entries), iou_val, matched


# -----------------------------------------------
# 1. Load model
# -----------------------------------------------
print('Loading model...')
full_model  = init_detector(CONFIG_FILE, CHECKPOINT, device=DEVICE)
full_model.eval()
class_names = CLASS_NAMES if CLASS_NAMES else None
if not class_names:
    try:
        class_names = list(full_model.dataset_meta['classes'])
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        result = process_image(image_path, full_model, class_names,
                               GT_DIR, OUTPUT_DIR)
        if result is not False:
            n_pred, n_gt, iou_val, matched = result
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

# Xóa file tạm
if os.path.exists(TMP_PATH):
    os.remove(TMP_PATH)

# -----------------------------------------------
# 4. Summary
# -----------------------------------------------
n_with_gt  = len(iou_scores)
mean_iou   = float(np.mean(iou_scores)) if iou_scores else 0.0
match_rate = total_matched / n_with_gt * 100 if n_with_gt > 0 else 0.0

print('\n========== DONE ==========')
print('Processed  : {}/{}'.format(len(all_images) - len(failed), len(all_images)))
print('Resize to  : {}px (short side)'.format(TARGET_SIZE))
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
