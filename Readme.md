# 🧠 CBAM-Residual U-Net for Histopathology Image Segmentation

This project implements a **U-Net baseline** and an **optimized CBAM-Residual U-Net** for **cell / nuclei segmentation** in histopathological images using PyTorch.

The goal is to **compare standard U-Net vs attention + residual learning** for semantic segmentation, following ideas from the paper:

**"An Optimized Multi-Organ Cancer Cells Segmentation for Histopathological Images Based on CBAM-Residual U-Net"**

This repository is designed for **quick experimentation and demo purposes**.

---

## 📁 Project Structure

```
unet-cbam-residual-blocks/
│
├── models/
│   ├── unet_plain.py          # Standard U-Net
│   ├── resunet_cbam.py        # Residual U-Net with CBAM
│   └── cbam.py                # CBAM attention module
│
├── dataset.py                  # Dataset loader
├── train.py                    # Training script
├── inference.py                # Inference & visualization
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
```

**Activate it:**

- **Windows**
  ```bash
  venv\Scripts\activate
  ```

- **Linux / macOS**
  ```bash
  source venv/bin/activate
  ```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Format

Organize your dataset as:

```
data/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/
    ├── mask_001.png
    ├── mask_002.png
    └── ...
```

**Images:** RGB histopathology image patches

**Masks:** Binary masks
- White (1): cell / nucleus
- Black (0): background

---

## 🚀 Training

Run training:

```bash
python train.py
```

This will:
- Load dataset
- Train the model
- Save trained weights as `.pth` files

**Example outputs:**
- `unet_plain.pth`
- `cbam_resunet.pth`

> `.pth` files are the learned model parameters after training.

---

## 🔍 Inference & Visualization

Run inference:

```bash
python inference.py
```

This will:
- Load trained models
- Perform segmentation
- Display side-by-side comparison:
  - Input image
  - Ground truth (GT)
  - U-Net prediction
  - CBAM-Residual U-Net prediction

This is intended for **qualitative comparison and demo**.

---

## 📈 What This Model Does (and Does NOT Do)

### ✔️ Does:
- Segments cell / nucleus regions
- Learns spatial + channel attention (CBAM)
- Demonstrates improvement over baseline U-Net

### ❌ Does NOT:
- Diagnose cancer
- Classify benign vs malignant cells

> **This is a segmentation task only.**

---

## 🧪 Notes on Training

⚠️ **Training on the same dataset and testing on it may cause overfitting**

- For demos, this is acceptable
- For real evaluation, use:
  - Train / validation split
  - Dice score / IoU metrics

---

## 🧠 Key Concepts Used

- U-Net architecture
- Residual learning
- CBAM (Channel + Spatial Attention)
- Binary segmentation
- Dice / IoU metrics (optional)

---

## 📌 Intended Use

- Academic learning
- Paper reproduction (partial)
- Proof-of-concept demos
- Architecture comparison

---

## 👤 Author

Your Name

Demo implementation inspired by academic literature

---

## 🚀 Next Steps

If you want to extend this project:

- **Feature map extraction script**
- **Attention map visualization**
- **Presentation slides**
- **Dice / IoU metric code**

Just say the word! 🚀

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

Based on concepts from:
- "An Optimized Multi-Organ Cancer Cells Segmentation for Histopathological Images Based on CBAM-Residual U-Net"
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- CBAM: Convolutional Block Attention Module