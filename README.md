# MASR-DNet: A Deformable Multi-scale Spatial Context Attention Network for Aortic Valve Detection in CT and Cardiomegaly Localization in Chest X-ray Images
---

We propose MASR-DNet, a compact one-stage detector specifically engineered for cardiac structure localization in CT and chest X-ray imaging. This architecture achieves state-of-the-art accuracy while maintaining a constrained parameter count of 12.6M.

We introduce a suite of four integrated feature enhancement modules to resolve specific challenges in medical imaging. These include the Geometric Adaptive Deformable Convolution (GADC) for irregular boundaries, the Enhanced Atrous Spatial Pyramid Pooling (EASPP) for multi-scale context, the Multi-scale Structural Refinement (MSRF) for high-resolution detail, and the Hierarchical Spatial Attention (HSA) for positional dependencies.

We develop the Adaptive Similarity Contrastive Loss (ASCL) to improve feature discriminability between foreground and background. This loss function effectively mitigates the severe class imbalance inherent in volumetric cardiac CT datasets through a stabilized margin-based constraint.

The proposed framework underwent rigorous evaluation on the AICUP 2025 (Cardiac CT) and VinDr-CXR (Cardiomegaly) benchmarks. Experimental results demonstrate statistically significant performance gains over 17 state-of-the-art methods across all primary evaluation metrics.

---
# Architecture of the proposed MASR-DNet.  
<img width="1316" height="775" alt="image" src="https://github.com/user-attachments/assets/ef92d7ba-6d84-452b-81b7-225269ac7511" />

---
# Geometric Adaptive Deformable Convolution module
<img width="765" height="269" alt="image" src="https://github.com/user-attachments/assets/426dcf3b-ef4e-4f26-bf40-a96ea4bef1a6" />

---
# Architecture of the Enhanced Atrous Spatial Pyramid Pooling module
<img width="756" height="414" alt="image" src="https://github.com/user-attachments/assets/54962aec-8b96-47d3-bca8-cf0dfdfe4e30" />

---
# Architecture of the Multi-Scale Refinement Fusion module
<img width="721" height="125" alt="image" src="https://github.com/user-attachments/assets/583d267c-b8c7-44c1-bc82-1aa2fb304832" />

---
# Architecture of the Hierarchical Spatial Attention module
<img width="736" height="525" alt="image" src="https://github.com/user-attachments/assets/1be47891-08e6-4edb-b37d-75b491db82e6" />

---
# Adaptive Similarity Contrastive Loss
<img width="629" height="112" alt="image" src="https://github.com/user-attachments/assets/37f99f2e-d0f1-44c5-95cf-1fdfd9ad36da" />

---
# Learnable branch weight dynamics in HSA
<img width="728" height="825" alt="image" src="https://github.com/user-attachments/assets/a9dacf94-c591-43b1-984f-9975908f2a0c" />

---
# Dataset aortic valve (AICUP 2025)
<img width="751" height="588" alt="image" src="https://github.com/user-attachments/assets/904c64f2-e758-4be3-a4ae-49eb258ed856" />

---
# Dataset cardiomegaly (VinDr-CXR)
<img width="752" height="419" alt="image" src="https://github.com/user-attachments/assets/1eef8822-f8cd-4645-9f58-ea7cfed2b97c" />

---
# Training
Hyperparameters:
* Epochs: 200
* Batch size: 16
* Pre-train: None

| Dataset   | mAP@50 | mAP@50:95 |
|-----------|--------|----------|
| AICUP25   | 96.4   | 68.2     |
| VinDr-CXR | 97.4   | 64.3     |

---
# AICUP25 Training
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/dc8510b2-aa17-4b0f-afd3-3143048fd0c4" />

<img width="1539" height="979" alt="image" src="https://github.com/user-attachments/assets/18b0fcaf-5959-4116-a4ee-638d6953e086" />

<img width="1490" height="809" alt="image" src="https://github.com/user-attachments/assets/26a5b8fe-ec92-4518-8d41-dfde368e77bf" />

# AICUP25 Testing
<img width="736" height="1010" alt="image" src="https://github.com/user-attachments/assets/7002fcd7-ccb6-4407-a930-2bfb66ec253f" />

----
# VinDr-CXR Training
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/cea398a7-0f56-4988-b778-158550ad7b53" />

<img width="1545" height="989" alt="image" src="https://github.com/user-attachments/assets/c0d5d45e-806f-4c95-ac22-df8a9e84d749" />

<img width="1499" height="997" alt="image" src="https://github.com/user-attachments/assets/7f2a62c8-0ec8-478b-a2ff-f2b73db074be" />

# VinDr-CXR Testing
<img width="741" height="1036" alt="image" src="https://github.com/user-attachments/assets/5a919598-73a3-4527-b891-0d03e08231a6" />


