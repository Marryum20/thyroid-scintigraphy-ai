# AI-Based Thyroid Scintigraphy Classification

A deep learning-based research project for the automated classification of thyroid scintigraphy images into five diagnostic categories.

## 🔬 Project Overview

This project investigates the application of **artificial intelligence and deep learning** to thyroid scintigraphy for computer-assisted diagnostic classification.

The study uses real-world nuclear medicine imaging data and evaluates deep learning models for distinguishing different thyroid conditions.

## 📊 Dataset

The research dataset contains **4,639 thyroid scintigraphy images** across five diagnostic categories:

* Diffuse Goiter
* Hyperfunctioning
* Hypofunctioning
* Multinodular Goiter (MNG)
* Normal

Patient-identifiable and restricted clinical data are not included in this repository.

## ⚙️ Methodology

The image-processing and deep-learning workflow includes:

**DICOM/PACS Data → Data Cleaning → Anonymization → Image Preprocessing → CLAHE Enhancement → Data Augmentation → Deep Learning → Classification → Model Evaluation**

### Image Processing

* DICOM/PACS data handling
* Data cleaning and organization
* Image anonymization
* CLAHE-based image enhancement
* Image resizing
* Data augmentation
* Image normalization

### Deep Learning

The project evaluates transfer-learning-based deep neural networks, including:

* ConvNeXt
* DenseNet
* ResNet

## 🧪 Technologies

* Python
* PyTorch
* Deep Learning
* Computer Vision
* Medical Image Processing
* DICOM
* PACS
* NumPy
* Pandas
* OpenCV

## 📈 Results

The best-performing model achieved approximately:

| Metric           |     Result |
| ---------------- | ---------: |
| Overall Accuracy | **89.15%** |
| Mean Sensitivity | **90.33%** |
| Mean Specificity | **97.22%** |
| Mean F1-Score    | **88.91%** |

Temporal validation using **258 previously unseen images** achieved approximately **91% accuracy**, indicating promising performance on recent clinical imaging data.

## 🎯 Research Objective

The primary objective is to investigate whether deep learning can provide reliable automated classification of thyroid scintigraphy images and potentially support nuclear medicine physicians in diagnostic decision-making.

## 📚 Research Status

This work forms part of my **MPhil research in Physics**, with a focus on Medical Physics, Nuclear Medicine, Medical Imaging, and Artificial Intelligence.

**Manuscript submitted for publication.**

## ⚠️ Data Privacy

This repository does not contain patient-identifiable information, clinical records, or restricted hospital imaging data.

The repository is intended to document the research methodology, software implementation, and shareable research materials.

---

### 👩‍🔬 Researcher

**Marryum Zia**
MPhil Physics | Medical Physics | Nuclear Medicine | AI & Medical Imaging
