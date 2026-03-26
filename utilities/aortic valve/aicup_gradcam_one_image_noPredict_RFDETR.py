# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import torch
import numpy as np
from rfdetr import RFDETRBase
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# fix: sua thang vao svd_on_activations
import pytorch_grad_cam.utils.svd_on_activations as svd_module
import pytorch_grad_cam.eigen_cam as eigen_cam_module

_original_get_2d_projection = svd_module.get_2d_projection

def _patched_get_2d_projection(activation_batch):
    if activation_batch is None:
        raise ValueError('activation_batch is None - hook failed')
    activation_batch = np.array(activation_batch, dtype=np.float32)
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for batch in activation_batch:
        reshaped = batch.reshape(batch.shape[0], -1).T
        reshaped -= reshaped.mean(axis=0)
        try:
            U, S, VT    = np.linalg.svd(reshaped, full_matrices=False)
            projection  = U[:, 0].reshape(batch.shape[1], batch.shape[2])
        except Exception:
            projection  = np.zeros((batch.shape[1], batch.shape[2]), dtype=np.float32)
        pmin, pmax = projection.min(), projection.max()
        if pmax > pmin:
            projection = (projection - pmin) / (pmax - pmin)
        projections.append(projection)
    return np.array(projections, dtype=np.float32)

svd_module.get_2d_projection       = _patched_get_2d_projection
eigen_cam_module.get_2d_projection = _patched_get_2d_projection

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_WEIGHTS = '/home/ai/Cardiomegaly/rfdetr/rfdetr_aicup/checkpoint0009.pth'
OUTPUT_FILE   = 'imgs/outputs/aicup/No_Predict/rfdetr_cam.png'

# Ảnh đơn cần visualize
IMAGE_PATH    = 'imgs/aicup/patient0051_0284.png'
GT_PATH       = 'imgs/aicup/ground_truth/patient0051_0284.txt'

GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2


# -----------------------------------------------
# Reshape transform cho DINOv2 windowed attention
# -----------------------------------------------
def dinov2_reshape_transform(tensor, height=None, width=None):
    if tensor.dim() == 3:
        B, N, C = tensor.shape
        for skip in [0, 1, 5, 9, 17]:
            remaining = N - skip
            s = int(remaining ** 0.5)
            if s * s == remaining and remaining > 0:
                patch = tensor[:, skip:, :] if skip > 0 else tensor
                return patch.reshape(B, s, s, C).permute(0, 3, 1, 2).float()
        for batch_size in [1, 2, 4]:
            total = B * N
            if total % batch_size == 0:
                tokens_per = total // batch_size
                s = int(tokens_per ** 0.5)
                if s * s == tokens_per:
                    return tensor.reshape(batch_size, s, s, C).permute(0, 3, 1, 2).float()
        avg = tensor.mean(dim=0, keepdim=True)
        s   = int(N ** 0.5)
        if s * s < N:
            s = s + 1
        pad = s * s - N
        if pad > 0:
            avg = torch.nn.functional.pad(avg, (0, 0, 0, pad))
        return avg.reshape(1, s, s, C).permute(0, 3, 1, 2).float()
    return tensor.float()


# -----------------------------------------------
# Wrapper
# -----------------------------------------------
class RFDETRWrapper(torch.nn.Module):
    def __init__(self, rfdetr_model):
        super().__init__()
        self.lwdetr   = rfdetr_model.model.model
        self.joiner   = self.lwdetr.backbone
        self.backbone = self.joiner[0]
        self.dinov2   = self.backbone.encoder
        self.encoder  = self.dinov2.encoder
        self.inner    = self.encoder.encoder

    def forward(self, x):
        x       = x.float()
        emb_out = self.encoder.embeddings(x)
        enc_out = self.inner(emb_out)
        hidden  = enc_out.last_hidden_state.float()
        B, N, C = hidden.shape
        for skip in [1, 5, 9, 17]:
            remaining = N - skip
            s = int(remaining ** 0.5)
            if s * s == remaining:
                return hidden[:, skip:, :].float()
        return hidden.float()


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
print('Loading RF-DETR model...')
rfdetr = RFDETRBase(
    num_classes=1,
    pretrain_weights=MODEL_WEIGHTS
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lwdetr = rfdetr.model.model
lwdetr.eval()
lwdetr.to(device)

wrapped = RFDETRWrapper(rfdetr)
wrapped.eval()
wrapped.to(device)

target_layers = [wrapped.inner.layer[-1].mlp]
print('Target layer: inner.layer[-1].mlp')

means = np.array(rfdetr.means, dtype=np.float32) * 255.0
stds  = np.array(rfdetr.stds,  dtype=np.float32) * 255.0
print('Model loaded:', MODEL_WEIGHTS)
print('Device:', device)

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
img_norm     = (img_rgb.astype(np.float32) - means) / stds
input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

# -----------------------------------------------
# 4. EigenCAM
# -----------------------------------------------
with EigenCAM(model=wrapped, target_layers=target_layers,
              reshape_transform=dinov2_reshape_transform) as cam:
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
        eigen_smooth=True,
        aug_smooth=False
    )

heatmap = grayscale_cam[0].astype(np.float32)
heatmap = cv2.resize(heatmap, (W, H))
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
