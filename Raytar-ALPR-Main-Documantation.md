# Iranian License Plate Detection & Recognition — Project Report


**Deliverables:** presentation notebook  (with outputs) + this report  
**Scope:** Phase 1 (detection) → Phase 2 (recognition) → Phase 3 (end-to-end) → Phase 4 (analysis)

---

## 1. Executive summary

We built a **two-stage Automatic License Plate Recognition (ALPR)** system for **Iranian (Persian) plates**:

1. **Detection (LPD):** find the plate bounding box in a full scene image using Ultralytics YOLO.
2. **Recognition (LPR):** read the cropped plate as a fixed 8-character string `DD L DDD DD` using multi-head CNNs.

**Final recommended stack**

| Stage | Model | Checkpoint file | Headline metric |
|---|---|---|---|
| Detection | **YOLO26n** | `yolo26.pt` | live mAP50-95 **0.804**, recall **0.967** |
| Recognition | **ScratchPlateCNN** | `scratch_lpr_cnn.pt` | plate acc **92.33%** (CER **1.63%**) |
| Recognition baseline | EfficientNet-B0 | `lpr_efficientnet_b0_best.pt` | plate acc **87.48%** |
| OCR baseline | EasyOCR (fa/en), 100-img subset | plate acc **0.00%**, CER **99.75%** |

**Main lesson:** a small plate-aware CNN beats both a larger ImageNet backbone and generic OCR on this task. Operational details (Drive checkpoints, label cleaning, matching preprocess to the checkpoint) mattered as much as architecture choice.

---


### 2.1 One-time setup for a new machine / reviewer


1. Upload the six `.pt` files to Google Drive (same Google account you use for Colab), with the exact names above.  
   Recommended path: `My Drive/ALPR_checkpoints/`.
2. (Optional) Upload `LPD.zip` and `LPR.zip` to the same Drive if you want live validation plots and recognition metrics. Without the zips, the notebook still loads models and shows our **recorded** result tables.

### 2.2 What the notebook does when you run it

1. Installs packages (`ultralytics`, `torch`, etc.).
2. Mounts Google Drive and **copies** the six checkpoints into `/content/weights/`.
3. With `SKIP_DET_TRAIN = True`, `SKIP_SCRATCH_TRAIN = True`, `SKIP_ENET_TRAIN = True` (defaults): **loads** pretrained models instead of retraining for hours.
4. If dataset zips are on Drive, copies/unzips them and runs live eval + demos.
5. If datasets are missing, still prints analysis from our recorded GPU metrics.

Leave the `SKIP_*` flags as `True` for presentation / TA review. Set them to `False` only if you intentionally want to retrain.


---

## 3. Related work and papers that shaped our design

We did not invent two-stage ALPR from scratch. The design follows recent Iranian and general ALPR literature, then maps it onto Ultralytics YOLO + PyTorch multi-head CNNs that we could train on Colab, as the project itself suggested, we just wanted to make sure the idea is accurate in the real world.

### 3.1 Core Iranian / ALPR papers

| Paper | Venue / year | Detection | Recognition | What we took |
|---|---|---|---|---|
| **Tourani et al.**, “A Robust Deep Learning Approach for Automatic Iranian Vehicle License Plate Detection and Recognition for Surveillance Systems” | IEEE Access, 2020 | YOLOv3-style detector | Character / plate recognizer cascade | **Two-stage real-time ALPR** for Iranian plates; detector first, then read the crop |
| **Hatami et al.**, Iranian plate ALPR with lightweight detection + sequence model | Scientia Iranica, 2024 | YOLOv4-tiny | CRNN + CTC | Tiny YOLO is enough for plates; **sequence recognition** beats naive per-char OCR; motivated our “small model + structure” bias |
| **IR-LPR** (Iranian License Plate Recognition benchmarks / datasets, ~2022 line of work) | — | Often YOLOv5 | YOLO / Detectron-style or CNN heads | Confirms **YOLO family** works well on Iranian plates at high mAP when fine-tuned |
| **DILP** and related ICRoM / newer Iranian ALPR systems | ~2024 | YOLOv10-class detectors | CNN–BiLSTM–CTC style | Keep the detector family **modern** (we compared YOLOv8, YOLO11, YOLO12, and YOLO26) |



---

## 4. Problem statement and design

### 4.1 Why two stages?

Following Tourani et al. (2020), Hatami et al. (2024), IR-LPR, and DILP-style systems, we separate:

- **Where is the plate?** → object detection (small object, variable scene)
- **What does it say?** → structured recognition (fixed layout)

End-to-end OCR on full frames is brittle: wrong scale, clutter, and no plate grammar.

### 4.2 Iranian plate grammar

Private plates follow:

```text
 positions:  0  1  2  3  4  5  6  7
 content:    D  D  L  D  D  D  D  D
```

Example: `98ع89511`. Digits may appear as ASCII or Persian `۰–۹` in the CSV. The letter slot uses a closed Persian alphabet (21 classes in our scratch checkpoint).

### 4.3 Datasets

| Dataset | Role | Size (approx.) | Notes |
|---|---|---|---|
| **LPD** | Detection | ~3.8 GB zip | YOLO labels `class cx cy w h`; we used ~85/15 train/val |
| **LPR** | Recognition | ~256 MB zip | `valid_samples.csv` + `detections/*.jpg`; ~24.5k usable rows → 90/10 split (`SEED=42`) |

Drive `gdown` was rate-limited often; we switched to **Drive mount + `cp`** (see §2).

---

## 5. Phase 1 — License plate detection

### 5.1 Models we trained

We saved **four** detectors with **exact Drive filenames**:

| File | Base | Training notes |
|---|---|---|
| `lpd_yolov8n_best.pt` | YOLOv8n | Fair recipe, 20 epochs |
| `lpd_yolo11n_best.pt` | YOLO11n | Fair recipe, 20 epochs |
| `lpd_yolo12n_best.pt` | YOLO12n | Fair recipe, 20 epochs |
| `yolo26.pt` | YOLO26n | **Selected for the pipeline** (best mAP50-95 / recall) |

### 5.2 Fair training recipe (v8 / v11 / v12)

Pinned so the comparison is honest:

| Hyperparameter | Value | Rationale |
|---|---|---|
| epochs | 20 (GPU) | Enough to fine-tune 1-class plates from COCO |
| imgsz | 640 | YOLO default; balances small-plate detail vs VRAM |
| batch | 16 | Fits Tesla T4 for nano models |
| optimizer | SGD | Ultralytics detection default |
| lr0 / lrf | 0.01 / 0.01 | Cosine decay down to `lr0 * lrf` |
| momentum | 0.937 | YOLO default |
| weight_decay | 5e-4 | Light regularization |
| cos_lr | True | Smoother late training |
| patience | 8 | Early stopping on fitness |
| pretrained | True | COCO → plate transfer |
| seed | 42 | Reproducibility |
| augments | Ultralytics mosaic / HSV / fliplr | Defaults; works well for plates |

### 5.3 Metrics and results

Numbers below come from the presentation notebook run (live Ultralytics `val` on this split: **937 images / 1076 instances**), plus the metrics stored in the checkpoints from GPU training.

**Live validation (this notebook)**

| Model | Precision | Recall | mAP50 | mAP50-95 | Val time (s) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.946 | 0.951 | 0.978 | 0.763 | ~193 |
| YOLO11n | 0.953 | 0.943 | 0.974 | 0.764 | ~172 |
| YOLO12n | 0.944 | 0.957 | 0.981 | 0.760 | ~246 |
| **YOLO26n** | **0.963** | **0.967** | **0.989** | **0.804** | **~162** |

**Recorded metrics inside the checkpoints (GPU training)**

| Model | Precision | Recall | mAP50 | mAP50-95 | Epochs |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.939 | 0.935 | 0.968 | 0.748 | 20 |
| YOLO11n | 0.948 | 0.931 | 0.968 | 0.746 | 20 |
| YOLO12n | 0.946 | 0.930 | 0.967 | 0.749 | 20 |
| **YOLO26n** | 0.944 | **0.968** | **0.986** | **0.777** | 50 |

Both tables agree: YOLO26n is clearly ahead.

### 5.4 How we chose the winner

- Primary key: **mAP50-95** across **all four** detectors.
- Tie-break / secondary: **recall** (missed plates kill end-to-end).
- **YOLO26n wins** on live val (mAP50-95 **0.804**, recall **0.967**, mAP50 **0.989**) and on recorded training metrics.
- YOLOv8n / 11n / 12n cluster tightly around mAP50-95 ≈ 0.76 live; YOLO26n jumps to **0.804**.
- End-to-end therefore uses **`yolo26.pt`** + ScratchPlateCNN.

### 5.5 Qualitative behavior

On validation scenes the detectors generally:

- find plates on dark / light cars,
- keep reasonable boxes at moderate distance,
- occasionally loosen the box on tilted / blurry plates (hurts recognition digits at the edges).

---

## 6. Phase 2 — License plate recognition

### 6.1 Task formulation

We treat recognition as **eight classification heads**:

- positions `{0,1,3,4,5,6,7}` → 10-way digit softmax  
- position `2` → 21-way letter softmax  

Loss = sum of cross-entropies (EfficientNet used label smoothing 0.1).

Metrics:

- **Plate accuracy** — entire string correct (strict; primary)
- **CER** — character error rate
- **Per-position accuracy** — which slot fails
- **Latency** — ms / image

### 6.2 Label cleaning (critical)

Raw CSV issues we fixed:

| Issue | Fix |
|---|---|
| `الف` multi-glyph token | replace with `ا` |
| Zero-width joiners `\u200d` | strip |
| Persian digits `۰–۹` | map to ASCII `0–9` |
| Non length-8 labels | drop |
| Letter `ا` after normalize not in 21-class alphabet | filter rows (scratch alphabet has no `ا`) |

Without this cleaning, training crashes (`KeyError`) or silently learns garbage.

### 6.3 Model I — Scratch residual CNN

**Architecture**

- Stem: Conv–BN–ReLU–Pool ×2 (3→32→64)
- Residual blocks + deeper 128-channel stage
- AdaptiveAvgPool → **LayerNorm(128)** → Dropout(0.35)
- 7 digit heads + 1 letter head
- ~**474,875** parameters

**Preprocess (must match checkpoint `config`)**

- Resize **64×256**
- Normalize mean/std **0.5**

**Training**

| Item | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Schedule | CosineAnnealingLR |
| Grad clip | 5.0 |
| Early stopping | patience 5 |
| Epochs ran | 25 |
| Batch | 64 ; we also eval’d with 32 |
| Train / test | 21,168 / 2,351 |

**Handoff metrics (test)**

| Metric | Value |
|---|---:|
| Char accuracy | **98.19%** |
| Full-plate accuracy | **91.79%** |
| Inference | ~0.27 ms/img (batch 64, MPS) |

**Our Colab reload / eval of the same weights**

| Metric | Value |
|---|---:|
| Plate accuracy | **92.33%** |
| CER | **1.63%** |
| ~ms/img | **~3.0** (includes loader overhead on T4 path) |

Slight differences vs handoff come from filtering / loader settings; both confirm **~92%** plate accuracy.

Checkpoint format:

```text
scratch_lpr_cnn.pt → {model_state_dict, config, history, num_params}
```

### 6.4 Model II — EfficientNet-B0 multi-head

**Architecture**

- ImageNet EfficientNet-B0 backbone (classifier replaced by Identity)
- Shared FC 1280→512 + BN + ReLU + Dropout(0.4)
- 8 typed heads (digit vs letter)
- ~**4.7M** trainable params

**Tricks**

- Dual LR: backbone `1e-5`, heads/shared `1e-3`
- Label smoothing 0.1
- Input **224×224**, ImageNet mean/std
- Early stopping patience 5; best at epoch **20** in our run

**Result**

| Metric | Value |
|---|---:|
| Plate accuracy | **87.48%** |
| CER | **~2.39%** |
| Checkpoint | `lpr_efficientnet_b0_best.pt` → `{model, acc, epoch}` |

### 6.5 EasyOCR baseline

Zero-shot EasyOCR (`fa`/`en`) on a **100-image** random test subset from our final notebook run:

| Metric | Value |
|---|---:|
| Plate accuracy | **0.00%** |
| CER | **99.75%** |
| Latency | **~2596 ms/img** (CPU) |

Generic OCR does not know Iranian plate length / grammar and often emits wrong-length strings. It is a useful **negative baseline**, not a competitor.

### 6.6 Comparison table (final notebook)

| Model | Plate acc % | CER % | Params | ms/img (this run) | Trained on LPR? |
|---|---:|---:|---:|---:|:---:|
| **ScratchPlateCNN** | **92.33** | **1.63** | 0.47M | ~19 | Yes |
| EfficientNet-B0 | 87.48 | 2.39 | 4.7M | ~62 | Yes |
| EasyOCR (100 imgs) | 0.00 | 99.75 | — | ~2596 | No |

LPR split used in the final notebook: **24,561** usable plates → **22,104** train / **2,457** test (`SEED=42`).

**Winner for Phase 3:** ScratchPlateCNN.

### 6.7 Error analysis

Frequent mistakes observed:

- Digit swaps **`2↔3`**, **`0↔1`**, especially near the plate end
- Letter confusions with similar glyphs (**`س↔ص`**, etc.)
- EfficientNet sometimes collapses several positions at once on hard crops
- EasyOCR: length errors, mixed scripts, unusable without heavy post-rules

Letter slot (index 2) is usually the hardest structurally; digit slots are high-90s when the crop is clean.

### 6.8 Why scratch beats EfficientNet here

1. Crops are already plate-aligned; a huge backbone is overkill.
2. **64×256** matches plate aspect ratio better than 224×224.
3. Typed heads encode Iranian grammar.
4. Fewer parameters → less overfitting on a moderate dataset.

EfficientNet remains valuable as a transfer-learning baseline and sanity check.

### 6.9 Trial-and-error (Phase 2)

1. **`KeyError: ا`** after `الف→ا` because alphabet has 21 letters without `ا` → filter.
2. Loading friend weights into wrong class / wrong key (`CustomPlateCNN` + `state["model"]`) → must use **`ScratchPlateCNN` + `model_state_dict`**.
3. Wrong preprocess (ImageNet normalize on scratch) silently hurts accuracy → always read **`config`** from the checkpoint.
4. `EPOCHS_LPR` undefined when skipping train cells → define defaults / load path first.
5. DataLoader `num_workers` shutdown noise on Colab → use `num_workers=0` for presentation.
6. Comparing OCR on full 2.4k test set is slow → subset for demos; CNNs still eval on full test when possible.

---

## 7. Phase 3 — End-to-end system

### 7.1 Pipeline

`LicensePlateSystem`:

1. YOLO `predict` with `conf=0.25`
2. Crop each `xyxy` box from the original BGR image
3. Apply recognizer preprocess (64×256, mean 0.5)
4. Argmax over 8 heads → decode string
5. Draw box + text for demos; report latency

**Default stack:** YOLO26n (`yolo26.pt`) + ScratchPlateCNN (`scratch_lpr_cnn.pt`)

### 7.2 Observed behavior

- Mean latency on demo images is typically on the order of tens–hundreds of ms / image on T4 (detect dominates).
- Failure modes: loose boxes → truncated digits; missed plates; multi-plate frames need list handling; dirty letter crops.



---

## 8. Techniques and engineering practices we used

| Technique | Where | Why |
|---|---|---|
| Transfer learning (COCO / ImageNet) | YOLO, EfficientNet | Faster convergence, better features |
| Rank detectors by mAP50-95 | Phase 1 | YOLO26n selected for end-to-end |
| Multi-head classification | Phase 2 | Fits fixed Iranian layout better than free OCR |
| Residual + BN + Dropout + LayerNorm | Scratch CNN | Stable training; assignment-friendly |
| Dual LR + label smoothing | EfficientNet | Preserve backbone, calibrate heads |
| Grad clipping (5.0) | Both CNNs | Avoid exploding updates |
| Early stopping | Both CNNs / YOLO patience | Save best weights |
| Label normalization | LPR CSV | Remove `الف` / ZWJ / Persian digits traps |
| Drive checkpoint discipline | All phases | Survive Colab disconnects |


---

---

## 9. Conclusion

We delivered a complete Iranian ALPR stack, guided by Tourani / Hatami / IR-LPR / DILP-style two-stage designs:

- **Four** YOLO detectors compared on the same val split; **YOLO26n** wins clearly (live mAP50-95 **0.804**, recall **0.967**) and is used end-to-end.
- **Two** recognition models, where a **~0.47M scratch CNN (~92% plate acc)** outperformed **EfficientNet-B0 (~87%)** and crushed **EasyOCR**.
- An **end-to-end** demo path and a reproducible **Git code + Drive weights** workflow.

The project reinforces a practical CV lesson: **match the model to the structure of the data**. Iranian plates are a detection problem plus a tiny grammar-constrained recognition problem — not a generic OCR problem.

---

## Appendix A — References (papers & techniques)

1. **A. Tourani et al.**, “A Robust Deep Learning Approach for Automatic Iranian Vehicle License Plate Detection and Recognition for Surveillance Systems,” *IEEE Access*, 2020.  
   → Two-stage Iranian ALPR; YOLO-style detection then recognition.

2. **M. Hatami et al.**, Iranian license-plate detection/recognition with lightweight YOLO and CRNN+CTC, *Scientia Iranica*, 2024.  
   → Tiny detectors + sequence recognition; motivated our preference for small, structured models.

3. **IR-LPR** and related Iranian LPR dataset / benchmark work (~2022).  
   → YOLO family performs strongly on Iranian plates when fine-tuned.

4. **DILP** and related newer Iranian ALPR systems (e.g. ICRoM 2024 line; YOLOv10-class detectors + CNN–BiLSTM–CTC).  
   → Keep detector baselines modern (YOLOv8 / 11 / 12 / 26 in our runs).

5. Ultralytics **YOLOv8 / YOLO11 / YOLO12** documentation and ALPR application papers using YOLOv8 cascades.  
   → Practical training recipe (SGD, cosine LR, mosaic) and cascade > classical OCR.

6. **LPRNet** and multi-head plate CNN literature.  
   → Fixed-length multi-head classification for plate strings.

7. **CRNN + CTC** sequence recognition literature (as used in many LPR systems including Hatami et al.).  
   → Cited as a strong alternative path we did not ship in the final multi-head assignment setup.

8. **EasyOCR** / **PaddleOCR** engine comparison practice in OCR surveys.  
   → Zero-shot baseline showing generic OCR fails on our Iranian crops without plate-specific training.

9. **EfficientNet** (Tan & Le) ImageNet transfer learning.  
   → Backbone for our Phase-2 fine-tune baseline.

| Source | Idea we borrowed in code |
|---|---|
| Tourani et al., 2020 | Two-stage real-time Iranian ALPR |
| Hatami et al., 2024 | Tiny YOLO + structured recognition |
| IR-LPR | YOLO strong on Iranian plates |
| DILP / newer YOLO ALPR | Modern YOLO family comparison |
| LPRNet / multi-head CNNs | Prefer plate-trained heads over classical OCR |
| EasyOCR comparisons | Baseline only, not the main recognizer |

## Appendix B — Checkpoint quick facts

| File | Type | Key fields / metrics |
|---|---|---|
| `lpd_yolov8n_best.pt` | Ultralytics | mAP50-95≈0.748; SGD 20ep |
| `lpd_yolo11n_best.pt` | Ultralytics | mAP50-95≈0.747 |
| `lpd_yolo12n_best.pt` | Ultralytics | live mAP50-95≈0.760 |
| `yolo26.pt` | Ultralytics | live mAP50-95≈**0.804** (**pipeline detector**) |
| `scratch_lpr_cnn.pt` | PyTorch | `model_state_dict` + `config`; ~92% plate |
| `lpr_efficientnet_b0_best.pt` | PyTorch | `model` + `acc≈87.48` + `epoch` |


