# AI-Based Thyroid Scintigraphy Classification

A deep learning-based research project for the automated classification of thyroid scintigraphy images into five diagnostic categories, with **Grad-CAM explainability** and a **Streamlit-based AI-assisted clinical demonstration interface**.

---

## 🔬 Project Overview

This project investigates the application of **Artificial Intelligence (AI), deep learning, and medical image processing** to thyroid scintigraphy for computer-assisted classification.

The research uses real-world nuclear medicine imaging data and evaluates multiple transfer-learning-based deep learning architectures for distinguishing different thyroid scintigraphic patterns.

The project extends beyond model development to include:

- DICOM/PACS data processing
- Data cleaning and anonymization
- Image preprocessing and enhancement
- Deep learning-based classification
- Statistical model evaluation
- Grad-CAM-based explainability
- A Streamlit-based clinical demonstration application
- Future external and real-time clinical validation

The overall objective is to develop an **AI-assisted tool that can support nuclear medicine physicians**, rather than replace expert clinical judgment.

---

## 📊 Dataset

The research dataset contains **4,639 thyroid scintigraphy images** distributed across five diagnostic categories:

| Class |
|---|
| Diffuse Goiter |
| Hyperfunctioning |
| Hypofunctioning |
| Multinodular Goiter (MNG) |
| Normal |

The dataset was derived from real-world nuclear medicine imaging data.

> **Privacy:** Patient-identifiable information, restricted clinical records, PACS credentials, and private hospital imaging data are not included in this repository.

---

## ⚙️ Research Workflow

The overall research pipeline is:

**DICOM/PACS Data → Data Cleaning → Anonymization → Data Labeling → DICOM-to-PNG Conversion → Image Preprocessing → CLAHE Enhancement → Data Augmentation → Deep Learning → Model Evaluation → Grad-CAM Explainability → Clinical Demonstration**

---

## 🖼️ Image Processing

The preprocessing workflow includes:

- DICOM/PACS data handling
- Data extraction and organization
- Data cleaning and quality control
- DICOM anonymization
- Image labeling and dataset organization
- DICOM-to-PNG conversion
- Intensity normalization
- CLAHE-based contrast enhancement
- Image resizing to **320 × 320 pixels**
- Data augmentation
- Preparation of training, validation, and test datasets

---

## 🧠 Deep Learning Models

Four transfer-learning-based architectures are evaluated:

- **ConvNeXt Tiny**
- **ConvNeXt Small**
- **DenseNet-201**
- **ResNet-50**

The models are implemented using **PyTorch** and **timm**.

Model performance is evaluated using classification and clinical performance metrics, including:

- Accuracy
- Sensitivity
- Specificity
- Precision / PPV
- NPV
- F1-score
- ROC-AUC
- Confusion matrices
- Confidence intervals

---

## 🔎 Explainable AI — Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is incorporated to provide visual explanations of model predictions.

Grad-CAM generates heatmaps highlighting image regions that contribute to the model's classification decision.

This explainability component is intended to:

- Improve interpretability of deep learning predictions
- Provide visual insight into model decision-making
- Support qualitative assessment by researchers and clinicians
- Investigate whether model attention is focused on clinically relevant regions

Grad-CAM is an **interpretability tool** and should not be considered independent clinical evidence.

The implementation is available in:

```text
explainability/
├── README.md
└── gradcam.py
