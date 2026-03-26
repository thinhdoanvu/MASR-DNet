# encoding: utf-8
import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmdet.apis import init_detector, inference_detector


# -----------------------------------------------
# CONFIG
# -----------------------------------------------
#CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/detr-1/config_aicup25_detr.py'
#CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/detr-1/epoch_3.pth'
#OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/Gradcam/DETR'

CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/dino-1/config_aicup25_dino.py'
CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/dino-1/best_coco_bbox_mAP_epoch_2.pth'
OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/Gradcam/DINO'

#CONFIG_FILE = '/home/ai/mmdetection3x/checkpoints/swinB-1/config_aicup25_swinB.py'
#CHECKPOINT  = '/home/ai/mmdetection3x/checkpoints/swinB-1/epoch_4.pth'
#OUTPUT_DIR  = '/home/ai/mmdetection3x/imgs/outputs/cardiomegaly/Gradcam/SWINB'

INPUT_DIR    = '/home/ai/mmdetection3x/imgs/cardiomegaly'
GT_DIR       = '/home/ai/mmdetection3x/imgs/cardiomegaly/ground_truth'
DEVICE       = 'cuda:0'
SCORE_THRESH = 0.3
IMG_EXTS     = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

CLASS_NAMES  = ['Cardiomegaly']
TARGET_CLASS = 0
OUTPUT_SIZE  = 480

PRED_COLOR    = (0, 220, 0)
GT_COLOR      = (220, 0, 0)
BOX_THICKNESS = 5
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 2
FONT_THICK    = 4


# -----------------------------------------------
# Swin reshape transform
# -----------------------------------------------
def swin_reshape_transform(tensor, height=None, width=None):
    """
    Swin output: [B, H*W, C] -> [B, C, H, W]
    Ho tro ca truong hop H != W (non-square images)
    """
    B, HW, C = tensor.shape

    # truong hop 1: chinh vuong
    s = int(HW ** 0.5)
    if s * s == HW:
        return tensor.reshape(B, s, s, C).permute(0, 3, 1, 2)

    # truong hop 2: non-square - tim H, W gan dung nhat
    # dua vao so luong tokens de suy ra H, W
    best = None
    best_diff = float('inf')
    for h in range(1, HW + 1):
        if HW % h == 0:
            w    = HW // h
            diff = abs(h - w)
            if diff < best_diff:
                best_diff = diff
                best = (h, w)
    if best is not None:
        h, w = best
        return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)

    # fallback: pad ve so chinh vuong
    s   = int(HW ** 0.5) + 1
    pad = s * s - HW
    tensor_padded = torch.nn.functional.pad(
        tensor.permute(0, 2, 1), (0, pad)
    ).permute(0, 2, 1)
    return tensor_padded.reshape(B, s, s, C).permute(0, 3, 1, 2)

# -----------------------------------------------
# Wrapper: backbone + neck
# -----------------------------------------------
class BackboneNeckWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.has_neck = hasattr(model, 'neck') and model.neck is not None
        if self.has_neck:
            self.neck = model.neck

    def forward(self, x):
        feat = self.backbone(x)
        if self.has_neck:
            feat = self.neck(feat)
            return feat[0]
        if isinstance(feat, (list, tuple)):
            return feat[-1]
        return feat


# -----------------------------------------------
# Auto-select target_layers
# -----------------------------------------------
def get_target_layers(model, wrapped):
    backbone      = model.backbone
    backbone_type = type(backbone).__name__
    print('Backbone type:', backbone_type)
    if hasattr(backbone, 'layer4'):
        print('Target layer: backbone.layer4[-1]')
        return [wrapped.backbone.layer4[-1]], None
    if hasattr(backbone, 'stages'):
        print('Target layer: backbone.stages[-1].blocks[-1].norm2')
        return [wrapped.backbone.stages[-1].blocks[-1].norm2], swin_reshape_transform
    if hasattr(backbone, 'layers'):
        print('Target layer: backbone.layers[-1].blocks[-1].norm2')
        return [wrapped.backbone.layers[-1].blocks[-1].norm2], swin_reshape_transform
    raise ValueError('Cannot find target layer for: ' + backbone_type)


# -----------------------------------------------
# Helper: resize giu ti le
# -----------------------------------------------
def resize_keep_ratio(img, target_width):
    if target_width is None:
        return img
    h, w  = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


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
# Helper: draw GT boxes (red)
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
# Helper: save 4-panel figure (optional)
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
# Helper: process single image
# -----------------------------------------------
def process_image(image_path, wrapped, full_model, cam, class_names, gt_dir, output_dir):
    base = os.path.splitext(os.path.basename(image_path))[0]

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print('  [SKIP] Cannot read:', image_path)
        return False

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W      = img_rgb.shape[:2]

    # -----------------------------------------------
    # Resize ve size co dinh GIU TI LE cho EigenCAM
    # Swin/DETR can input size on dinh de window attention khong bi soc
    # -----------------------------------------------
    BACKBONE_SIZE = 1024  # thu voi 512 hoac 800 neu van bi soc

    scale    = BACKBONE_SIZE / max(H, W)
    new_H    = int(H * scale)
    new_W    = int(W * scale)
    img_resized = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)

    # pad ve BACKBONE_SIZE x BACKBONE_SIZE
    canvas = np.zeros((BACKBONE_SIZE, BACKBONE_SIZE, 3), dtype=np.uint8)
    canvas[:new_H, :new_W] = img_resized

    # img_float de overlay heatmap - dung anh da pad
    img_float = np.float32(canvas) / 255.0

    # preprocess
    mean = np.array([123.675, 116.28,  103.53], dtype=np.float32)
    std  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
    img_norm     = (canvas.astype(np.float32) - mean) / std
    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    print('  [SIZE] original={}x{} -> padded={}x{}'.format(W, H, BACKBONE_SIZE, BACKBONE_SIZE))

    # EigenCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                        eigen_smooth=True, aug_smooth=False)
    heatmap      = grayscale_cam[0].astype(np.float32)  # [BACKBONE_SIZE, BACKBONE_SIZE]
    heatmap      = np.clip(heatmap, 0, 1)

    # crop phan pad ve kich thuoc anh da resize
    heatmap_cropped  = heatmap[:new_H, :new_W]
    img_float_cropped = img_float[:new_H, :new_W]

    # resize heatmap va anh ve kich thuoc goc de ve box dung toa do
    heatmap_orig  = cv2.resize(heatmap_cropped,  (W, H), interpolation=cv2.INTER_LINEAR)
    img_float_orig = cv2.resize(img_float_cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    cam_only_rgb = show_cam_on_image(img_float_orig, heatmap_orig, use_rgb=True)

    # inference tren anh goc (full resolution)
    det_result = inference_detector(full_model, image_path)
    pred       = det_result.pred_instances
    keep       = (pred.scores > SCORE_THRESH) & (pred.labels == TARGET_CLASS)
    boxes_np   = pred.bboxes[keep].cpu().numpy().astype(int)
    scores_np  = pred.scores[keep].cpu().numpy()
    labels_np  = pred.labels[keep].cpu().numpy()

    if len(boxes_np) > 1:
        best_idx  = np.argmax(scores_np)
        boxes_np  = boxes_np[best_idx:best_idx+1]
        scores_np = scores_np[best_idx:best_idx+1]
        labels_np = labels_np[best_idx:best_idx+1]

    print('  [PRED] kept={}'.format(len(boxes_np)))

    # GT
    gt_entries = load_gt_for_image(base, gt_dir)
    has_gt     = len(gt_entries) > 0
    print('  [GT] {}'.format('found {} box(es)'.format(len(gt_entries)) if has_gt else 'no GT file'))

    # ve box tren anh goc (toa do box la full resolution)
    cam_full_rgb = draw_pred_boxes(cam_only_rgb, boxes_np, scores_np, labels_np, class_names)
    if has_gt:
        cam_full_rgb = draw_gt_boxes(cam_full_rgb, gt_entries, class_names)

    # resize output cho paper
    cam_full_bgr = cv2.cvtColor(cam_full_rgb, cv2.COLOR_RGB2BGR)
    cam_full_bgr = resize_keep_ratio(cam_full_bgr, OUTPUT_SIZE)

    cv2.imwrite(os.path.join(output_dir, base + '_heatmap_full.png'), cam_full_bgr)

    return len(boxes_np), len(gt_entries)

# -----------------------------------------------
# 1. Load model
# -----------------------------------------------
print('Loading model...')
full_model = init_detector(CONFIG_FILE, CHECKPOINT, device=DEVICE)
full_model.eval()

wrapped = BackboneNeckWrapper(full_model).to(DEVICE)
wrapped.eval()

class_names = CLASS_NAMES if CLASS_NAMES else None
if not class_names:
    try:
        class_names = list(full_model.dataset_meta['classes'])
    except Exception:
        class_names = None

target_layers, reshape_transform = get_target_layers(full_model, wrapped)

print('Class names  :', class_names)
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
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# 3. Process all
# -----------------------------------------------
total_pred = 0
total_gt   = 0
failed     = []

with EigenCAM(model=wrapped, target_layers=target_layers,
              reshape_transform=reshape_transform) as cam:
    for idx, image_path in enumerate(all_images):
        print('[{}/{}] {}'.format(idx + 1, len(all_images), os.path.basename(image_path)))
        try:
            result = process_image(image_path, wrapped, full_model, cam,
                                   class_names, GT_DIR, OUTPUT_DIR)
            if result is not False:
                n_pred, n_gt = result
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
