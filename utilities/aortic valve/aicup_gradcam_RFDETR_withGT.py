# encoding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
        # batch: [C, H, W]
        reshaped = batch.reshape(batch.shape[0], -1).T  # [H*W, C]
        reshaped -= reshaped.mean(axis=0)
        try:
            U, S, VT = np.linalg.svd(reshaped, full_matrices=False)
            projection = U[:, 0].reshape(batch.shape[1], batch.shape[2])
        except Exception:
            projection = np.zeros((batch.shape[1], batch.shape[2]), dtype=np.float32)
        # normalize to [0, 1]
        pmin, pmax = projection.min(), projection.max()
        if pmax > pmin:
            projection = (projection - pmin) / (pmax - pmin)
        projections.append(projection)
    return np.array(projections, dtype=np.float32)

svd_module.get_2d_projection           = _patched_get_2d_projection
eigen_cam_module.get_2d_projection     = _patched_get_2d_projection


# -----------------------------------------------
# CONFIG
# -----------------------------------------------
MODEL_WEIGHTS = '/home/ai/Cardiomegaly/rfdetr/checkpoints_aicup/checkpoint0009.pth'

INPUT_DIR     = 'imgs/aicup'
GT_DIR        = 'imgs/aicup/ground_truth'
OUTPUT_DIR    = 'imgs/outputs/aicup/Gradcam/RFDETR_GT'
SCORE_THRESH  = 0.3
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
CLASS_NAMES   = ['aortic_valve']

PRED_COLOR    = (0, 255, 0)
GT_COLOR      = (255, 0, 0)
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICK    = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# Reshape transform cho DINOv2 windowed attention
# -----------------------------------------------
def dinov2_reshape_transform(tensor, height=None, width=None):
    #print('    [reshape] shape={} dtype={}'.format(tensor.shape, tensor.dtype))
    if tensor.dim() == 3:
        B, N, C = tensor.shape

        # thu cac truong hop skip CLS + register tokens
        for skip in [0, 1, 5, 9, 17]:
            remaining = N - skip
            s = int(remaining ** 0.5)
            if s * s == remaining and remaining > 0:
                patch = tensor[:, skip:, :] if skip > 0 else tensor
                return patch.reshape(B, s, s, C).permute(0, 3, 1, 2).float()

        # windowed attention: B*num_windows, window_tokens, C
        # gop lai
        for batch_size in [1, 2, 4]:
            total = B * N
            if total % batch_size == 0:
                tokens_per = total // batch_size
                s = int(tokens_per ** 0.5)
                if s * s == tokens_per:
                    return tensor.reshape(batch_size, s, s, C).permute(0, 3, 1, 2).float()

        # fallback: average qua B dimension
        avg = tensor.mean(dim=0, keepdim=True)  # [1, N, C]
        s   = int(N ** 0.5)
        if s * s < N:
            s = s + 1
        # pad neu can
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
        self.encoder  = self.dinov2.encoder        # WindowedDinov2WithRegistersBackbone
        self.inner    = self.encoder.encoder       # WindowedDinov2WithRegistersEncoder

    def forward(self, x):
        x       = x.float()
        emb_out = self.encoder.embeddings(x)
        enc_out = self.inner(emb_out)
        hidden  = enc_out.last_hidden_state.float()  # [B, N_tokens, C]

        # bo CLS + register tokens, chi giu patch tokens
        B, N, C = hidden.shape
        for skip in [1, 5, 9, 17]:
            remaining = N - skip
            s = int(remaining ** 0.5)
            if s * s == remaining:
                return hidden[:, skip:, :].float()

        return hidden.float()


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
print('Loading RF-DETR model...')
rfdetr = RFDETRBase(pretrain_weights=MODEL_WEIGHTS)

lwdetr = rfdetr.model.model
lwdetr.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lwdetr.to(device)

wrapped = RFDETRWrapper(rfdetr)
wrapped.eval()
wrapped.to(device)

# target layer: mlp cua layer cuoi
target_layers = [wrapped.inner.layer[-1].mlp]
print('Target layer: inner.layer[-1].mlp')

means = np.array(rfdetr.means, dtype=np.float32) * 255.0
stds  = np.array(rfdetr.stds,  dtype=np.float32) * 255.0

try:
    class_names = list(rfdetr.class_names.values()) if isinstance(rfdetr.class_names, dict) else rfdetr.class_names
except Exception:
    class_names = CLASS_NAMES
print('Class names:', class_names)
print('Device:', device)


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

with EigenCAM(model=wrapped, target_layers=target_layers,
              reshape_transform=dinov2_reshape_transform) as cam:

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

            # preprocess
            img_norm     = (img_rgb.astype(np.float32) - means) / stds
            input_tensor = torch.from_numpy(img_norm.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # EigenCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                                eigen_smooth=True, aug_smooth=False)
            heatmap      = grayscale_cam[0].astype(np.float32)
            heatmap      = cv2.resize(heatmap, (W, H))
            heatmap      = np.clip(heatmap, 0, 1)
            cam_only_rgb = show_cam_on_image(img_float, heatmap, use_rgb=True)

            # RF-DETR inference
            pil_img   = Image.fromarray(img_rgb)
            det       = rfdetr.predict(pil_img, threshold=SCORE_THRESH)
            if len(det) > 0:
                boxes_np  = det.xyxy.astype(int)
                scores_np = det.confidence
                labels_np = det.class_id
            else:
                boxes_np  = np.zeros((0, 4), dtype=int)
                scores_np = np.array([])
                labels_np = np.array([])

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