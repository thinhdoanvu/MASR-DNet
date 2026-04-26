# MASR-DNet

**A Deformable Multi-scale Spatial Context Attention Network for Aortic Valve Detection in CT and Cardiomegaly Localization in Chest X-ray Images**

> Paper under review at *Biomedical Signal Processing and Control*

---

## Overview

MASR-DNet is a compact one-stage object detection framework designed for cardiac structure localization across two imaging modalities — contrast-enhanced CT and chest X-ray. Built on a YOLOv9-M backbone, it integrates five task-specific modules to address class imbalance, calcification artifacts, and cross-modality resolution heterogeneity.

| Dataset | mAP@50 | mAP@50:95 | Params |
|---|---|---|---|
| Cardiac CT (AI CUP 2025) | 96.23 ± 0.32% | 68.03 ± 0.31% | 14.5M |
| Cardiomegaly (VinDr-CXR) | 97.27 ± 0.36% | 63.93 ± 0.32% | 12.6M |

Both results are the highest among 17 compared baselines (two-stage, transformer-based, and one-stage detectors).

---

## Architecture

| Module | Role |
|---|---|
| **GADC** — Geometry-Aware Deformable Convolution | Adaptive spatial sampling for irregular cardiac boundaries |
| **EASPP** — Enhanced Atrous Spatial Pyramid Pooling | Multi-scale context aggregation with strip pooling |
| **MSRF** — Multi-scale Residual Fusion | Cross-resolution feature alignment via PixelShuffle |
| **HSA** — Hybrid Spatial Attention | Coordinate + dilated branch attention for positional focus |
| **ASCL** — Adaptive Supervised Contrastive Loss | Foreground–background separation under extreme class imbalance |

---

## Installation

```bash
git clone https://github.com/thinhdoanvu/MASR-DNet.git
cd MASR-DNet
pip install -e .
```

---

## Datasets

### Cardiac CT — AI CUP 2025
- 100 patients, 33,483 axial slices
- Foreground ratio: 16.9% (severe class imbalance)
- Split: 4:1:5 (train / val / test) at patient level
- Download: https://tbrain.trendmicro.com.tw/Competitions/Details/42

```yaml
# aortic_valve.yaml
path: /your/path/AICUP25
train: train/images
val:   val/images
nc: 1
names: ['aortic_valve']
```

### Cardiomegaly — VinDr-CXR
- 2,625 CXR images with cardiomegaly findings
- Split: 1,837 / 262 / 526 (train / val / test)
- Bounding boxes merged via WBF (IoU = 0.5) → 2,721 GT boxes
- Download: https://physionet.org/content/vindr-cxr/1.0.0/

```yaml
# cardiomegaly.yaml
path: /your/path/Cardiomegaly
train: images/train
val:   images/val
nc: 1
names: ['Cardiomegaly']
```

---

## Training

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("cfg/models/v8/masr_dnet.yaml")
    model.train(
        data="aortic_valve.yaml",   # or cardiomegaly.yaml
        epochs=200,
        batch=64,
        imgsz=640,
        optimizer="SGD",
        lr0=1e-4,
        weight_decay=0.01,
        warmup_epochs=10,
        patience=50,
        box=12,
        cls=0.5,
        close_mosaic=15,
    )
```

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("weights/masr_dnet_ct.pt")   # or masr_dnet_cxr.pt
results = model.predict("your_image.png", imgsz=640)
results[0].show()
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{MASRDNet2026,
  author  = {Vu Thinh Doan and Jonnagaddala Jitendra and Thi Thu Thuy Pham and Dinh HUng Nguyen and Hong-Jie Dai},
  title   = {MASR-DNet: A Deformable Multi-scale Spatial Context Attention Network
             for Aortic Valve Detection in CT and Cardiomegaly Localization
             in Chest X-ray Images},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.
