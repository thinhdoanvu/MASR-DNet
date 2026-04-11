# MASR-DNet: A Deformable Multi-scale Spatial Context Attention Network for Aortic Valve Detection in CT and Cardiomegaly Localization in Chest X-ray Images
---

We propose MASR-DNet, a compact one-stage detector specifically engineered for cardiac structure localization in CT and chest X-ray imaging. This architecture achieves state-of-the-art accuracy while maintaining a constrained parameter count of 12.6M.

We introduce a suite of four integrated feature enhancement modules to resolve specific challenges in medical imaging. These include the Geometric Adaptive Deformable Convolution (GADC) for irregular boundaries, the Enhanced Atrous Spatial Pyramid Pooling (EASPP) for multi-scale context, the Multi-scale Structural Refinement (MSRF) for high-resolution detail, and the Hierarchical Spatial Attention (HSA) for positional dependencies.

We develop the Adaptive Similarity Contrastive Loss (ASCL) to improve feature discriminability between foreground and background. This loss function effectively mitigates the severe class imbalance inherent in volumetric cardiac CT datasets through a stabilized margin-based constraint.

The proposed framework underwent rigorous evaluation on the AICUP 2025 (Cardiac CT) and VinDr-CXR (Cardiomegaly) benchmarks. Experimental results demonstrate statistically significant performance gains over 17 state-of-the-art methods across all primary evaluation metrics.

---
# How to install.  
```
pip instal -e .
```

# Cardiac CT dataset
```
path: /home/ai/AICUP25
train: train/images
val: val/images
nc: 1
names: [
        'aortic_valve'
       ]
```
# VinDr-CXR dataset
```
path: /home/ai/Cardiomegaly
train: images/train
val: images/val
nc: 1
names: [
          'Cardiomegaly'
       ]
```
# Hyper parameters
```
from ultralytics import YOLO, RTDETR

if __name__ == "__main__":
    model = YOLO("cfg/models/abl/v4.yaml")
    model.train(
         data="../AICUP25/aortic_valve.yaml",
         epochs=200,
         batch=64,
         imgsz=640,
         optimizer="SGD",
         lr0=1e-4,
         weight_decay=0.01,
         warmup_epochs=10,
         patience = 50,
         box = 12,
         cls = 0.5,
         close_mosaic=15,
    )
```
