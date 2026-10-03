# Sustainable Waste Management System (SWMS) – China Area

**Intelligent Multi-Stage YOLO-based Waste Visual Classification System**

[![GitHub](https://img.shields.io/badge/GitHub-ADRIANHHALIM-blue?logo=github)](https://github.com/ADRIANHHALIM)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-red)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A complete computer-vision pipeline for **Sustainable Waste Management** tailored to **China-area waste classification rules**.  
> Features a two-stage hierarchy (Router → Specialist models), edge-ready exports (ONNX / TFLite), high mAP performance, and research context for IoT + Blockchain integration.

---

## Key Results

| Model                  | mAP50  | mAP50-95 | Precision | Recall | F1     |
|------------------------|--------|----------|-----------|--------|--------|
| **Stage 1 Router v2**  | 98.47  | 98.13    | 92.86     | 96.55  | 0.9467 |
| Stage 2 – Recyclable   | 99.27  | 99.23    | 98.45     | 98.10  | 0.9827 |
| Stage 2 – FoodWaste    | 99.50  | 99.28    | 96.69     | 98.67  | 0.9767 |
| Stage 2 – Others       | 98.55  | 98.17    | 98.76     | 96.30  | 0.9752 |
| Stage 2 – Harmful      | 94.19  | 93.52    | 95.33     | 87.08  | 0.9102 |

---

## System Architecture

The system uses a **hierarchical (two-stage) detection pipeline**:

1. **Stage 1 – Router** (`stage1_router_v2`)  
   Classifies waste into 4 high-level categories based on China municipal waste sorting rules:
   - `FoodWaste` (厨余垃圾)
   - `Harmful` (有害垃圾)
   - `Recyclable` (可回收物)
   - `Others` (其他垃圾)

2. **Stage 2 – Specialist Models**  
   Fine-grained classification inside each category:
   - **Recyclable**: Cardboard, GlassBottle, GlassContainer, Metal, Paper, PlasticBottle, PlasticPackaging, Textile
   - **FoodWaste**, **Harmful**, **Others**: domain-specific classes

Models are trained with **YOLOv8l** and exported to:
- `.pt` (PyTorch)
- `.onnx` (ONNX Runtime)
- `.tflite` (TensorFlow Lite – edge devices)

---

## Repository Structure

```
Sustainable-Waste-Management-System-China/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes          # Git LFS tracking for large model files
├── configs/
│   ├── stage1_router_args.yaml
│   └── stage1_router_v2_args.yaml
├── results/
│   └── stage1_router_v2_results.csv
├── metrics/
│   └── test_mAP_from_running_data.csv
├── notebooks/              # Training & inference notebooks
├── models/                 # Place model weights here (see below)
└── docs/
    └── (architecture diagrams & evaluation plots)
```

---

## Model Weights & Datasets

Because of GitHub file size limits, the large artifacts are hosted externally:

### Model Weights (recommended)
Download from the original Google Drive folder or Kaggle:
- `stage1_router_v2` (.pt / .onnx / .tflite)
- `stage2_food`, `stage2_harmful`, `stage2_others`, `stage2_recyclable`

### Datasets (~1.9 GB each)
- `yolo_dataset_stage1.zip`
- `stage1_final.zip`
- `yolo_flat.zip`

**Google Drive folder**: [Sustainable Waste Management System](https://drive.google.com/drive/folders/1zQn22eDQQ-L_pE1KHedT3KQVAOMSLPFG)

After downloading, place models into the `models/` directory following the structure used during training.

> **Tip**: For production / edge deployment use the `.tflite` or `.onnx` versions.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ADRIANHHALIM/Sustainable-Waste-Management-System-China.git
cd Sustainable-Waste-Management-System-China

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Git LFS if you plan to track large model files
git lfs install
```

### Inference example (Ultralytics)

```python
from ultralytics import YOLO

# Stage 1 Router
router = YOLO("models/stage1_router_v2.pt")
results = router("path/to/waste_image.jpg")

# Then route to the corresponding Stage-2 model based on the predicted class
```

---

## Training Configuration (Stage 1 Router v2)

- Base model: `yolov8l.pt`
- Epochs: 120
- Batch size: 32
- Image size: 640
- Optimizer: SGD
- Patience: 30
- Device: CUDA

Full configuration is available in `configs/stage1_router_v2_args.yaml`.

---

## China Waste Classification Context

This project follows the common **four-category municipal solid waste classification** used in many Chinese cities (Shanghai, Beijing, Ningbo, etc.):

| Category (EN)   | Category (ZH)   | Typical Items                          |
|-----------------|-----------------|----------------------------------------|
| FoodWaste       | 厨余垃圾         | Leftovers, fruit peels, vegetable scraps |
| Recyclable      | 可回收物         | Paper, plastic bottles, metal, glass, textile |
| Harmful         | 有害垃圾         | Batteries, medicine, fluorescent lamps, paint |
| Others          | 其他垃圾         | Contaminated items, ceramic, cigarette butts |

The hierarchical design improves both accuracy and interpretability compared to a single flat multi-class detector.

---

## Research Context

Related work and future directions include:

- IoT smart bins (ultrasonic + weight + gas sensors)
- Sensor conflict consensus / anomaly detection
- Time-to-Full (TTF) forecasting
- Blockchain for transparent waste tracking & incentive mechanisms
- Federated Learning + Digital Twin frameworks (SABER)

See the accompanying research documents in the original Google Drive for more details.

---

## Kaggle

A corresponding **Kaggle Dataset + Notebook** will be published containing:

- Cleaned YOLO-format dataset
- Training notebooks
- Inference & evaluation notebooks

(Link will be added here once published)

---

## Author

**Adrian Halim**  
Full Stack Developer & AI/ML Engineer  
GitHub: [ADRIANHHALIM](https://github.com/ADRIANHHALIM)

---

## License

This project is released under the [MIT License](LICENSE).
