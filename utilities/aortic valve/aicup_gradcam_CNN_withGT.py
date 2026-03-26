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
#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/cascade_rcnn-1/config_aicup25_cascade_rcnn.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/cascade_rcnn-1/aicup_epoch_10.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/CascadeRCNN_GT'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/dynamic_rcnn-1/config_aicup25_dynamic_faster_RCNN.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/dynamic_rcnn-1/aicup_epoch_11.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/DYNAMIC_RCNN_GT'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/faster_rcnn-1/config_aicup25_fasterRCNN.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/faster_rcnn-1/aicup_epoch_12.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/fasterRCNN_GT'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/fcos-1/config_aicup25_fcos.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/fcos-1/aicup_epoch_77.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/FCOS_GT'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/retina-1/config_aicup25_retina.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/retina-1/aicup_epoch_37.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/RETINA_GT'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/sparse-1/config_aicup25_sparse.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/sparse-1/aicup_epoch_10.pth'
#OUTPUT_DIR   = '/home/ai/mmdetection3x/imgs/SPARSE_GT'


INPUT_DIR    = '/home/ai/mmdetection3x/imgs'
GT_DIR       = '/home/ai/mmdetection3x/imgs/ground_truth'   # moi anh 1 file txt
DEVICE       = 'cuda:0'
SCORE_THRESH = 0.3
IMG_EXTS     = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
CLASS_NAMES  = ['aortic_valve']

PRED_COLOR    = (0, 255, 0)
GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICK    = 1


# -----------------------------------------------
# Wrapper
# -----------------------------------------------
class BackboneNeckWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck     = model.neck

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        return feat[0]


# -----------------------------------------------
# Helper: load GT cho 1 anh
# file: GT_DIR/patient0051_0222.txt
# format moi dong: image_name class cx cy w h  (normalized)
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
            # format: class cx cy w h  (NO image_name prefix!)
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
# Helper: process single image
# -----------------------------------------------
def process_image(image_path, wrapped, full_model, cam, class_names, gt_dir, output_dir):
    base = os.path.splitext(os.path.basename(image_path))[0]

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print('  [SKIP] Cannot read:', image_path)
        return False

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    H, W      = img_rgb.shape[:2]

    # preprocess
    mean = np.array([123.675, 116.28,  103.53], dtype=np.float32)
    std  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
    img_norm     = (img_rgb.astype(np.float32) - mean) / std
    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # EigenCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                        eigen_smooth=True, aug_smooth=False)
    heatmap      = cv2.resize(grayscale_cam[0], (W, H))
    heatmap      = np.clip(heatmap, 0, 1)
    cam_only_rgb = show_cam_on_image(img_float, heatmap, use_rgb=True)

    # inference
    det_result = inference_detector(full_model, image_path)
    pred       = det_result.pred_instances
    keep       = pred.scores > SCORE_THRESH
    boxes_np   = pred.bboxes[keep].cpu().numpy().astype(int)
    scores_np  = pred.scores[keep].cpu().numpy()
    labels_np  = pred.labels[keep].cpu().numpy()

    # GT - load truc tiep tu file txt tuong ung
    gt_entries = load_gt_for_image(base, gt_dir)
    has_gt     = len(gt_entries) > 0
    print('  [GT] {}'.format('found {} box(es)'.format(len(gt_entries)) if has_gt else 'no GT file'))

    # build images
    orig_gt_rgb   = draw_gt_boxes(img_rgb, gt_entries, class_names) if has_gt else img_rgb.copy()
    orig_pred_rgb = draw_pred_boxes(img_rgb, boxes_np, scores_np, labels_np, class_names)
    cam_full_rgb  = draw_pred_boxes(cam_only_rgb, boxes_np, scores_np, labels_np, class_names)
    if has_gt:
        cam_full_rgb = draw_gt_boxes(cam_full_rgb, gt_entries, class_names)

    # save individual
    #cv2.imwrite(os.path.join(output_dir, base + '_orig_gt.png'), cv2.cvtColor(orig_gt_rgb,   cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join(output_dir, base + '_orig_pred.png'), cv2.cvtColor(orig_pred_rgb, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join(output_dir, base + '_heatmap.png'), cv2.cvtColor(cam_only_rgb,  cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, base + '_heatmap_full.png'), cv2.cvtColor(cam_full_rgb,  cv2.COLOR_RGB2BGR))

    # save panel
    #save_panel(orig_gt_rgb, orig_pred_rgb, cam_only_rgb, cam_full_rgb, os.path.join(output_dir, base + '_panel.png'))

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
total_pred = 0
total_gt   = 0
failed     = []

target_layers = [wrapped.backbone.layer4[-1]]

with EigenCAM(model=wrapped, target_layers=target_layers) as cam:
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