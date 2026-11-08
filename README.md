# Iranian License Plate Detection & Recognition (Raytar ALPR)

A robust, two-stage **Automatic License Plate Recognition (ALPR)** system tailored for **Iranian (Persian) license plates**. This repository contains trained models, Jupyter notebooks for training and evaluation, detailed documentation, and comparative performance analysis across multiple state-of-the-art architectures.

---

## 📌 Project Overview

Iranian license plates follow a strict structural grammar (`DD L DDD DD`), making end-to-end generic OCR brittle and inefficient. Following literature (e.g., Tourani et al., Hatami et al.), we decouple the problem into two specialized stages:
1. **License Plate Detection (LPD):** Locate the bounding box of the license plate within full scene images using Ultralytics YOLO models.
2. **License Plate Recognition (LPR):** Crop the plate and process it through a multi-head classification network to read the 8-character string (7 digits + 1 Persian letter).

---

## 📂 Repository Structure

```text
Raytar_ALPR/
├── InDepth-Compration-of-Phase2-models.pdf  # In-depth analysis report of Phase 2 models
├── Raytar-ALPR-Main-Documantation.md        # Comprehensive project documentation & report
├── Raytar-ALPR-Main-notebook.ipynb          # Main presentation & execution Jupyter notebook
├── Seprate-Pipeline-notebook.ipynb          # Separate pipeline testing notebook
└── trained_models/                          # Pre-trained model checkpoints
    ├── lpd_yolo11n_best.pt                  # YOLO11n plate detector
    ├── lpd_yolo12n_best.pt                  # YOLO12n plate detector
    ├── lpd_yolov8n_best.pt                  # YOLO8n plate detector
    ├── lpr_efficientnet_b0_best.pt          # EfficientNet-B0 recognition baseline
    ├── scratch_lpr_cnn.pt                   # Custom Scratch Residual CNN (Recommended)
    └── yolo26.pt                            # YOLO26n plate detector (Recommended)
```

---

## 🚀 Performance Summary

### 1. License Plate Detection (Phase 1)
We evaluated four YOLO nano detectors trained on the LPD dataset (20 epochs, 640px):

| Model | Checkpoint File | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | `lpd_yolov8n_best.pt` | 0.946 | 0.951 | 0.978 | 0.763 |
| YOLO11n | `lpd_yolo11n_best.pt` | 0.953 | 0.943 | 0.974 | 0.764 |
| YOLO12n | `lpd_yolo12n_best.pt` | 0.944 | 0.957 | 0.981 | 0.760 |
| **YOLO26n (Selected)** | **`yolo26.pt`** | **0.963** | **0.967** | **0.989** | **0.804** |

### 2. License Plate Recognition (Phase 2)
Recognition is formulated as 8 classification heads (7 digit slots + 1 Persian letter slot):

| Model | Checkpoint File | Plate Accuracy | CER (%) | Parameters | Notes |
|---|---|---:|---:|---:|---|
| **ScratchPlateCNN (Selected)** | **`scratch_lpr_cnn.pt`** | **92.33%** | **1.63%** | ~0.47M | Custom residual CNN (~64x256 input) |
| EfficientNet-B0 | `lpr_efficientnet_b0_best.pt` | 87.48% | 2.39% | ~4.7M | ImageNet transfer backbone |
| EasyOCR (Baseline) | — | 0.00% | 99.75% | — | Zero-shot generic OCR failure |

---

## 🚗 Iranian Plate Grammar

Private Iranian plates follow the format:
```text
Positions:  0  1  2  3  4  5  6  7
Content:    D  D  L  D  D  D  D  D
```
* **`D`**: Digit (`0–9`)
* **`L`**: Persian Alphabet Letter (e.g., `ب`, `ج`, `د`, `س`, `ص`, `ط`, `ق`, `ل`, `م`, `ن`, `و`, `ه`, `ی`, etc.)

---

## 💻 Getting Started & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ameernsr/license-plate-detection.git
   cd license-plate-detection
   ```

2. **Explore the Notebooks:**
   Open `Raytar-ALPR-Main-notebook.ipynb` in Google Colab or Jupyter Lab.

3. **Inference Pipeline:**
   The recommended end-to-end pipeline (`LicensePlateSystem`) combines **`yolo26.pt`** for detection and **`scratch_lpr_cnn.pt`** for recognition:
   - Detects plate bounding boxes (`conf=0.25`).
   - Crops and normalizes plate regions (`64×256`).
   - Predicts all 8 character slots simultaneously via multi-head classification.

---

## 📚 References & Literature

1. **A. Tourani et al.**, “A Robust Deep Learning Approach for Automatic Iranian Vehicle License Plate Detection and Recognition for Surveillance Systems,” *IEEE Access*, 2020.
2. **M. Hatami et al.**, Iranian license-plate detection/recognition with lightweight YOLO and CRNN+CTC, *Scientia Iranica*, 2024.
3. Ultralytics **YOLOv8 / YOLO11 / YOLO12 / YOLO26** documentation and detection benchmarks.
