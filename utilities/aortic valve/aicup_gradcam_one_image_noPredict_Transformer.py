# encoding: utf-8
import cv2
import torch
import numpy as np
import os
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mmdet.apis import init_detector

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/detr-1/config_aicup25_detr.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/detr-1/aicup_epoch_10.pth'
#OUTPUT_FILE  = '/home/ai/mmdetection3x/imgs/outputs/aicup/No_Predict/detr_cam.png'

#CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/dino_aicup/config_aicup25_dino.py'
#CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/dino_aicup/best_coco_bbox_mAP_epoch_14.pth'
#OUTPUT_FILE  = '/home/ai/mmdetection3x/imgs/outputs/aicup/No_Predict/dino_cam.png'

CONFIG_FILE  = '/home/ai/mmdetection3x/checkpoints/swinB-1/config_aicup25_swinB.py'
CHECKPOINT   = '/home/ai/mmdetection3x/checkpoints/swinB-1/aicup_epoch_14.pth'
OUTPUT_FILE  = '/home/ai/mmdetection3x/imgs/outputs/aicup/No_Predict/swinb_cam.png'

# Ảnh đơn cần visualize
IMAGE_PATH    = '/home/ai/mmdetection3x/imgs/patient0051_0284.png'
GT_PATH       = '/home/ai/mmdetection3x/imgs/aicup/ground_truth/patient0051_0284.txt'

DEVICE        = 'cuda:0'
GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2


# -----------------------------------------------
# Swin reshape transform
# -----------------------------------------------
def swin_reshape_transform(tensor, height=None, width=None):
    """Swin output: [B, H*W, C] -> [B, C, H, W]"""
    B, HW, C = tensor.shape
    h = w = int(HW ** 0.5)
    result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
    return result


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
        # Swin không có neck: trả về feature cuối
        if isinstance(feat, (list, tuple)):
            return feat[-1]
        return feat


# -----------------------------------------------
# Auto-select target_layers theo backbone
# -----------------------------------------------
def get_target_layers(model, wrapped):
    backbone      = model.backbone
    backbone_type = type(backbone).__name__
    print('Backbone type:', backbone_type)

    # ResNet: DETR, DINO
    if hasattr(backbone, 'layer4'):
        print('Target layer: backbone.layer4[-1]')
        return [wrapped.backbone.layer4[-1]], None

    # SwinTransformer MMDet: dùng 'stages'
    if hasattr(backbone, 'stages'):
        print('Target layer: backbone.stages[-1].blocks[-1].norm2')
        return [wrapped.backbone.stages[-1].blocks[-1].norm2], swin_reshape_transform

    # SwinTransformer HuggingFace: dùng 'layers'
    if hasattr(backbone, 'layers'):
        print('Target layer: backbone.layers[-1].blocks[-1].norm2')
        return [wrapped.backbone.layers[-1].blocks[-1].norm2], swin_reshape_transform

    raise ValueError('Cannot find target layer for backbone: ' + backbone_type)


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
print('Loading model...')
full_model = init_detector(CONFIG_FILE, CHECKPOINT, device=DEVICE)

print('--- Backbone attributes ---')
for name, module in full_model.backbone.named_children():
    print(name, ':', type(module).__name__)

full_model.eval()
wrapped = BackboneNeckWrapper(full_model).to(DEVICE)
wrapped.eval()
print('Model loaded:', CHECKPOINT)

# auto-select target_layers
target_layers, reshape_transform = get_target_layers(full_model, wrapped)

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
# 3. Preprocess
# -----------------------------------------------
mean = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
std  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
img_norm     = (img_rgb.astype(np.float32) - mean) / std
input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# -----------------------------------------------
# 4. EigenCAM
# -----------------------------------------------
with EigenCAM(model=wrapped, target_layers=target_layers,
              reshape_transform=reshape_transform) as cam:
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