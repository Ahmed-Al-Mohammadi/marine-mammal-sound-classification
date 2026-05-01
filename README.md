# 🐋 Marine Mammal Sound Classification

> AI system that identifies **47 marine mammal species** from audio recordings using Deep Learning & Classical ML — achieving **95%+ accuracy**

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-green)
![Classes](https://img.shields.io/badge/Classes-47_Species-purple)

---

## 📌 Overview

| Item | Detail |
|------|--------|
| Dataset | Watkins Marine Mammal Sound Database (WHOI) |
| Files | 15,234 WAV files — 47 species |
| Pipeline (DL) | WAV → Mel Spectrogram → CNN |
| Pipeline (ML) | WAV → Statistical Features (774) → ML Models |
| Hardware | RTX 3050 Ti (4GB VRAM) |

---

## 🏆 Results

| Model | Test Accuracy | F1 Score |
|-------|--------------|----------|
| **ResidualCNN** 🥇 | **95.67%** | **95.68%** |
| LightCNN | 95.54% | 95.54% |
| CustomCNN | 95.41% | 95.39% |
| SVM (RBF) | 91.86% | 91.69% |
| Extra Trees | 89.17% | 88.60% |
| Random Forest | 88.32% | 87.73% |

---

## 🧠 Models

### Deep Learning (PyTorch)
- **CustomCNN** — 4.9M parameters, 5 ConvBlocks + GAP
- **ResidualCNN** — 3.2M parameters, Skip Connections
- **LightCNN** — 145K parameters, Depthwise Separable Convolutions

### Machine Learning (sklearn + From Scratch)
Built **6 algorithms completely from scratch** (no sklearn):

| Algorithm | Built-in | From Scratch |
|-----------|----------|-------------|
| KNN | ✅ | ✅ |
| Naive Bayes | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ |
| Linear Regression | ✅ | ✅ |
| Decision Tree | ✅ | ✅ |
| Random Forest | ✅ | ✅ |

---

## 🔧 Pipeline
WAV Files (15,234)
↓
Mel Spectrogram (128 × 216)
↓
┌──────────────────┬──────────────────┐
│   DL Pipeline    │   ML Pipeline    │
│   3 CNN Models   │  774 Features    │
│   PyTorch + GPU  │  9 ML Models     │
└──────────────────┴──────────────────┘
↓
MLflow Tracking + SHAP Analysis
---

## 📊 Key Visualizations

| Mel Spectrograms | Training Curves |
|-----------------|-----------------|
| ![mel](figures/mel_spectrograms.png) | ![curves](figures/dl_training_curves.png) |

| Confusion Matrix | SHAP Analysis |
|-----------------|---------------|
| ![cm](figures/confusion_matrix_ResidualCNN.png) | ![shap](figures/shap_feature_importance.png) |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
# https://cis.whoi.edu/science/B/whalesounds/

# Run notebook
jupyter notebook MarineMammal_Classification.ipynb
```

---

## 📁 Project Structure
marine-mammal-sound-classification/
├── MarineMammal_Classification.ipynb  # Main notebook
├── class_labels_indices_whales.csv    # Species labels
├── requirements.txt
├── figures/                           # All visualizations
│   ├── mel_spectrograms.png
│   ├── dl_training_curves.png
│   ├── confusion_matrix_ResidualCNN.png
│   ├── shap_feature_importance.png
│   └── builtin_vs_scratch.png
└── README.md

---

## 🔍 Key Findings

- **High-frequency Mel bins (119-127)** are the most important features — biologically valid since dolphins & whales vocalize in ultrasonic ranges
- **ResidualCNN** outperforms all models due to skip connections preventing vanishing gradients
- **529x class imbalance** handled via WeightedRandomSampler
- Training time reduced from **300+ min → 40 sec/epoch** via Mel Spectrogram caching

---

## 📚 Dataset

**Watkins Marine Mammal Sound Database**
Woods Hole Oceanographic Institution (WHOI)
- ~2,000 unique recordings
- 60+ species
- Recordings from 1940s–2000s

🔗 [Dataset Link](https://cis.whoi.edu/science/B/whalesounds/)

---

**Ahmed Yasser** | CS450 Machine Learning | Modern Academy Maadi | 2026
