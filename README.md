# 🚁 Drone Human & Car Detection System
### Antlings Internship — AI/ML Technical Assessment

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv8n](https://img.shields.io/badge/Model-YOLOv8n-red.svg)](https://ultralytics.com)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://arxiv.org/abs/2110.06864)
[![Dataset](https://img.shields.io/badge/Dataset-VisDrone2019-green.svg)](https://github.com/VisDrone/VisDrone-Dataset)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20GPU-20BEFF.svg)](https://kaggle.com)
[![Seed](https://img.shields.io/badge/Seed-42-lightgrey.svg)]()

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Demo Video](#-demo-video)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Tasks Completed](#-tasks-completed)
- [How to Run on Kaggle](#-how-to-run-on-kaggle)
- [Training Configuration](#-training-configuration)
- [Evaluation Metrics](#-evaluation-metrics)
- [Strengths & Limitations](#-strengths--limitations)
- [References](#-references)

---

## 🎯 Overview

A complete **computer vision pipeline** for analyzing drone/aerial imagery from the [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) dataset.

**What it does:**
- Detects **humans** (pedestrians + people) and **cars** from drone images using fine-tuned **YOLOv8n**
- Draws **bounding boxes** with class label + confidence score on each detection
- Renders a live **count banner**: `HUMANS: X    CARS: Y    TOTAL: Z`
- Tracks objects across video frames using **ByteTrack** with persistent color-coded IDs
- Evaluates with **mAP@50, mAP@50:95, Precision, Recall, FPS**

**Tech stack:** `ultralytics` · `opencv-python` · `supervision` · `PyTorch` · Kaggle T4 GPU

---

## 🎬 Demo Video

📹 **[Watch Full Demo (3–5 min)](https://drive.google.com/file/d/12MPg6qipZPAWC_11r2IN6mcjQRPXPxIS/view?usp=sharing)**


---

## 📊 Results

> **Run the notebook on Kaggle to generate your numbers, then fill in this table.**

| Metric | Value |
|--------|-------|
| **mAP@50** | *(0.535)* |
| **mAP@50:95** | *(0.369)* |
| **Precision** | *(0.897)* |
| **Recall** | *(0.5)* |


*Validation protocol: `conf=0.001`, `iou=0.6` — standard mAP computation setting*

### Sample Outputs

<img width="1738" height="682" alt="image" src="https://github.com/user-attachments/assets/ece7242e-de57-4348-a695-bac7366db14e" />


<img width="1672" height="553" alt="image" src="https://github.com/user-attachments/assets/6f964734-13a9-4ff3-9fa2-d56da26074c0" />

---






---

## 📦 Dataset

**VisDrone2019-DET** — drone footage across 14 Chinese cities, varying altitudes and conditions.

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 6,471 | ✅ |
| Val | 548 | ✅ |
| Test-Dev | 1,610 | ❌ (no labels) |

**Source:** [Kaggle — banuprasadb/visdrone-dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

> The notebook **auto-downloads the dataset via Kaggle API** — no manual Add Data step needed.

### Raw Annotation Format

Each `.txt` file in `annotations/` has one object per line:
```
<x1>, <y1>, <width>, <height>, <score>, <category_id>, <truncation>, <occlusion>
```

### Category Mapping

The notebook collapses 12 VisDrone categories into 2 YOLO classes:

| VisDrone ID | Name | → YOLO Class | Label |
|---|---|---|---|
| 1 | pedestrian | → **0** | `human` |
| 2 | people | → **0** | `human` |
| 4 | car | → **1** | `car` |
| 0 | ignored | *skipped* | — |
| 11 | others | *skipped* | — |
| 3,5,6,7,8,9,10 | bicycle, van, truck, etc. | *skipped* | — |

Conversion function: `convert_visdrone_to_yolo()` — normalises `(x1,y1,w,h)` to `(cx,cy,w,h)` in `[0,1]`, clamps degenerate boxes (`w < 1e-4` or `h < 1e-4` are discarded), symlinks images to save disk space on Kaggle.

### Dataset Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Tiny objects** | Humans can be as small as 5–20px | Mosaic augmentation, 640px input |
| **High density** | 100+ objects per image in crowd scenes | Tuned NMS IoU=0.45 |
| **Scale variance** | Altitude changes cause 10× size range | `scale=0.5` augmentation |
| **Occlusion** | Heavy overlap in crowd scenes | ByteTrack low-conf tracklet rescue |
| **Class imbalance** | pedestrian >> car >> others | Focal loss in YOLO CE head |
| **Lighting variance** | Day / night / overcast | HSV jitter (`h=0.015, s=0.7, v=0.4`) |
| **Motion blur** | Fast drone movement | Random blur at data loading |

---

## 🏗️ Pipeline Architecture

```
VisDrone Raw .txt Annotations
              │
              ▼
┌─────────────────────────────────────────┐
│  Task 01 — Preprocessing & EDA         │
│                                         │
│  flexible_find_splits()                 │  Handles nested/flat VisDrone structures
│  diagnose_dataset()                     │  Prints full folder tree for debugging
│  parse_visdrone_annotation()            │  Parses CSV-style .txt, skips cat 0 & 11
│  convert_visdrone_to_yolo()             │  → normalised YOLO .txt labels
│  _parse_yolo_label_for_eda_local()      │  EDA scanning in YOLO format
│  draw_yolo_annotations()               │  GT visualization (YOLO format)
│  6-panel EDA figure                    │  distribution · density · bbox area
└──────────────┬──────────────────────────┘
               │  visdrone_yolo/train/images + labels
               │  visdrone_yolo/val/images + labels
               │  dataset.yaml
               ▼
┌─────────────────────────────────────────┐
│  Task 02 — YOLOv8n Fine-tuning         │
│                                         │
│  Base model  : yolov8n.pt (COCO)       │
│  Epochs      : 50                       │
│  Input size  : 640 × 640               │
│  Batch size  : 16                       │
│  Optimizer   : SGD (YOLO default)      │
│  LR schedule : Cosine + 3-ep warmup    │
│  Close mosaic: last 10 epochs          │
│  Conf thresh : 0.25 (training)         │
│  NMS IoU     : 0.45                    │
└──────────────┬──────────────────────────┘
               │  runs/visdrone_yolov8n/weights/best.pt
               ▼
┌─────────────────────────────────────────┐
│  Task 03 — Detection + Counting        │
│                                         │
│  detect_and_count(conf=0.25, iou=0.45) │
│  ┌──────────────────────────────────┐  │
│  │ class 0 → Human                  │  │
│  │   bbox color  : (57, 255, 20)    │  │  bright green BGR
│  │   label       : "Human 0.XX"     │  │
│  │ class 1 → Car                    │  │
│  │   bbox color  : (0, 140, 255)    │  │  orange BGR
│  │   label       : "Car 0.XX"       │  │
│  └──────────────────────────────────┘  │
│  Banner: "HUMANS: X    CARS: Y    TOTAL: Z"  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────────┐
       ▼                    ▼
┌────────────────┐   ┌──────────────────────┐
│   Task 04      │   │      Task 05         │
│  ByteTrack     │   │    Evaluation        │
│                │   │                      │
│ conf = 0.30    │   │ model.val()          │
│ iou  = 0.45    │   │   conf = 0.001       │
│ Kalman Filter  │   │   iou  = 0.6         │
│ Color per ID   │   │ FPS: 50-image bench  │
│ tracked MP4    │   │ Count histograms     │
│                │   │ GT vs Pred plots     │
└────────────────┘   └──────────────────────┘
```

---

## ✅ Tasks Completed

| # | Task | Key Functions / Output | Status |
|---|------|------------------------|--------|
| **01** | Dataset Understanding & Preprocessing | `flexible_find_splits()`, `parse_visdrone_annotation()`, `convert_visdrone_to_yolo()`, EDA 6-panel plot, GT annotation visualization | ✅ |
| **02** | Model Training (YOLOv8n) | `model.train()` — 50 epochs, 640px, mosaic+mixup+copy-paste, training curves | ✅ |
| **03** | Detection + Human Counting | `detect_and_count()` — green/orange bboxes, count banner overlay | ✅ |
| **04** | Object Tracking — *Bonus* | `model.track(tracker='bytetrack.yaml')` — color-coded track IDs, tracked MP4 | ✅ |
| **05** | Evaluation & Visualization | `model.val()` — mAP, P, R; FPS benchmark; count histograms; GT vs Pred | ✅ |

---

## 🚀 How to Run on Kaggle

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook** → upload `antlings_visdrone_assessment_final_.ipynb`
2. **Settings → Accelerator → GPU T4** *(mandatory — training takes ~2–3 h)*
3. **Do NOT click "Add Data"** — the notebook downloads via Kaggle API automatically
4. **Run All**

**What happens at the dataset cell:**
```
📥 Dataset not in /kaggle/input. Downloading via Kaggle API...
✅ Download complete: /kaggle/working/visdrone_raw
📂 Contents:
   📁 VisDrone2019-DET-train   (6471 items)
   📁 VisDrone2019-DET-val     (548 items)
   📁 VisDrone2019-DET-test-dev
```

**Dataset download fallback chain:**
1. Check `/kaggle/input/visdrone-dataset` (if added via UI)
2. `kaggle datasets download -d banuprasadb/visdrone-dataset --unzip`
3. `subprocess` + manual zip extraction

### Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/antlings-drone-detection.git
cd antlings-drone-detection
pip install -r requirements.txt

# Detect on a single image
python src/detect.py --source image.jpg --weights path/to/best.pt

# Detect on a folder of images
python src/detect.py --source images/ --weights path/to/best.pt

# Track on video with ByteTrack
python src/detect.py --source video.mp4 --weights path/to/best.pt --track
```

---

## ⚙️ Training Configuration

All values taken directly from the `model.train()` call in the notebook:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model` | `yolov8n.pt` | Pretrained on COCO — transfer learning |
| `epochs` | `50` | Sufficient for fine-tuning from COCO weights |
| `imgsz` | `640` | Standard VisDrone resolution |
| `batch` | `16` | Fits T4 GPU (15 GB VRAM) |
| `device` | `cuda` | T4 GPU on Kaggle |
| `cos_lr` | `True` | Cosine annealing LR schedule |
| `warmup_epochs` | `3` | Gradual LR warmup |
| `close_mosaic` | `10` | Disable mosaic last 10 epochs for stable convergence |
| `conf` | `0.25` | Inference confidence threshold |
| `iou` | `0.45` | NMS IoU threshold |
| `save_period` | `10` | Save checkpoint every 10 epochs |

**Augmentation values (exact):**

| Augmentation | Value | Effect |
|---|---|---|
| `mosaic` | `1.0` | 4-image tiling every batch — critical for small/dense objects |
| `mixup` | `0.1` | Blends 2 images — mild regularization |
| `copy_paste` | `0.1` | Pastes objects across images — small object diversity |
| `flipud` | `0.1` | Vertical flip — rare in real drone footage |
| `fliplr` | `0.5` | Horizontal flip — standard spatial invariance |
| `degrees` | `10.0` | Random rotation ±10° — drone tilt |
| `translate` | `0.1` | Random translation ±10% |
| `scale` | `0.5` | Random scale 50% — altitude-induced size variation |
| `hsv_h` | `0.015` | Hue jitter |
| `hsv_s` | `0.7` | Saturation jitter — handles overcast/sunny variation |
| `hsv_v` | `0.4` | Value jitter — handles day/night variation |

**Validation protocol (`model.val()`):**

| Parameter | Value |
|---|---|
| `conf` | `0.001` — low threshold to not miss any TP during AP sweep |
| `iou` | `0.6` — IoU matching threshold for TP/FP labeling |
| `imgsz` | `640` |
| `batch` | `16` |

---

## 📈 Evaluation Metrics

Metrics are computed on the **VisDrone val set (548 images)** using `model.val()`:

```
Metric          Variable    Definition
────────────────────────────────────────────────────────
Precision    →  mp          TP / (TP + FP)   — averaged over classes
Recall       →  mr          TP / (TP + FN)   — averaged over classes
mAP@50       →  map50       Average Precision at IoU=0.50
mAP@50:95    →  map5095     Mean AP over IoU thresholds 0.50→0.95 (step 0.05)
FPS          →  fps         50-image benchmark on T4 GPU (ms/frame reported too)
```

Per-class AP is also available via `val_results.box.ap_class_index`.

Counting analysis runs over **100 validation images** at `conf=0.25, iou=0.45` and reports per-image human/car count mean, median, and max.

---

## ⚡ Strengths & Limitations

### ✅ Strengths
- **Single-stage, end-to-end** — no two-step proposal overhead, fast inference
- **Mosaic at 1.0** gives strong recall on small objects packed in crowds
- **ByteTrack** (conf=0.3) rescues partially occluded tracks via low-confidence pass — no Re-ID overhead needed
- **Human counting logic is transparent**: `sum(class_id == 0)` per frame — easy to audit
- **Robust dataset finder** (`flexible_find_splits` + `diagnose_dataset`) handles any VisDrone folder structure from Kaggle
- **Runs at real-time speed** on GPU

### ⚠️ Limitations

| Issue | Root Cause | Impact |
|-------|-----------|--------|
| Humans < ~15px missed | 640px input + small anchor coverage | Undercounting in high-altitude shots |
| Dense crowd occlusion | NMS removes overlapping boxes | Human undercounting |
| YOLOv8n accuracy ceiling | Fewest parameters in YOLOv8 family | Lower mAP than YOLOv8m/l/x |
| Tracking IDs reset across non-consecutive frames | No global Re-ID | ID continuity lost between segments |
| Validation GT viz uses `annotations/` hardcoded path | GT vs Pred cell | Falls back to image-only if path missing |

### 🔧 Future Improvements

| Improvement | Expected Gain |
|---|---|
| Upgrade to YOLOv8m or YOLOv8l | +3–5% mAP |
| SAHI sliced inference (512px tiles) | +8–12% mAP on tiny objects |
| Increase input size to 1280px | Better small-object recall |
| Test-time augmentation (TTA) | +1–3% mAP |
| DeepSORT with Re-ID embeddings | Stable identity across occlusions |
| Train 100+ epochs | Better final convergence |
| P6 detection head | Dedicated small-object scale |

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [VisDrone Dataset — Zhu et al., 2018](https://arxiv.org/abs/1804.07437)
- [ByteTrack — Zhang et al., 2021](https://arxiv.org/abs/2110.06864)
- [SAHI — Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [Kaggle VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

---

## 👤 Author

**[Md Sahidul Islam Nisho]**
Antlings Internship Program — AI/ML Technical Assessment · May 2026

---
*YOLOv8n · ByteTrack · VisDrone2019 · Python · Kaggle GPU · Seed 42*
