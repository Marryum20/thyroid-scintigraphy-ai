# AI-Based Thyroid Scintigraphy Classification

A deep learning-based research project for the automated classification of thyroid scintigraphy images, with **Grad-CAM explainability and a Streamlit-based clinical demonstration interface**.

## 🔬 Project Overview

This project investigates the application of **artificial intelligence and deep learning** to thyroid scintigraphy for computer-assisted diagnostic classification.

The study uses real-world nuclear medicine imaging data and evaluates deep learning models for distinguishing different thyroid scintigraphic patterns.

In addition to model development and evaluation, the project includes an **explainability component using Grad-CAM** and a **prototype application for AI-assisted clinical implementation**.

The broader objective is to progress from retrospective model development toward **external and real-time clinical validation**, supporting the future integration of AI into nuclear medicine workflows.

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

**DICOM/PACS Data → Data Cleaning → Anonymization → Image Preprocessing → CLAHE Enhancement → Data Augmentation → Deep Learning → Classification → Evaluation → Explainability → Clinical Demonstration**

### Image Processing

* DICOM/PACS data handling
* Data cleaning and organization
* Image anonymization
* CLAHE-based image enhancement
* Image resizing to **320 × 320 pixels**
* Image normalization
* Data augmentation

### Deep Learning

The project evaluates transfer-learning-based deep neural networks, including:

* ConvNeXt Tiny
* ConvNeXt Small
* DenseNet-201
* ResNet-50

### Explainable AI

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is incorporated to visualize image regions contributing to model predictions.

This provides an interpretability layer that may help researchers and clinicians understand the visual features influencing AI classification.

Grad-CAM is intended as an **explainability aid**, not as independent clinical evidence.

### Clinical Demonstration Application

A **Streamlit-based prototype application** has been developed to demonstrate potential clinical implementation.

The application provides:

* DICOM image upload
* AI architecture selection
* Automated image preprocessing
* Thyroid scintigraphy classification
* Prediction confidence
* Class probability visualization
* Grad-CAM visualization
* AI-generated classification summary

The application is designed as a research prototype demonstrating how an AI model could be integrated into a nuclear medicine workflow.

## 🧪 Technologies

* Python
* PyTorch
* timm
* Streamlit
* Grad-CAM
* Deep Learning
* Computer Vision
* Medical Image Processing
* DICOM
* PACS
* NumPy
* Pandas
* OpenCV
* scikit-learn
* Matplotlib

## 📈 Results

### Model Performance

| Model          | Accuracy | 95% Confidence Interval |
| -------------- | -------: | ----------------------: |
| ConvNeXt Tiny  |  **92%** |             89.94–94.05 |
| DenseNet-201   |  **91%** |             88.76–93.11 |
| ConvNeXt Small |  **90%** |             87.92–92.43 |
| ResNet-50      |  **89%** |             85.43–90.38 |

Temporal validation using **258 previously unseen images** achieved approximately **91% accuracy** with the best-performing model, demonstrating promising performance on more recent clinical imaging data.

## 🎯 Research Objective

The primary objective is to investigate whether deep learning can provide reliable automated classification of thyroid scintigraphy images and potentially support **nuclear medicine physicians in diagnostic decision-making**.

The system is intended as an **AI-assisted tool rather than a replacement for expert clinical judgment**.

## 🏥 Clinical Implementation and Validation

A prototype clinical demonstration interface has been developed to explore the practical integration of the trained models into a nuclear medicine workflow.

The longer-term research direction includes:

* Integration of AI-assisted classification into clinical workflows
* External validation using data from independent nuclear medicine centers
* Real-time validation on newly acquired clinical scans
* Prospective evaluation of model performance
* Assessment of model robustness across different imaging conditions and patient populations
* Evaluation of Grad-CAM visualizations by nuclear medicine experts
* Collaboration with multiple nuclear medicine centers for multicenter validation

These steps are intended to assess whether the developed framework can maintain reliable performance beyond the original research dataset and move toward practical clinical implementation.

## 🔭 Future Development

Future work will focus on extending the framework from thyroid scintigraphy toward **AI-assisted analysis of other static nuclear medicine imaging studies**, subject to appropriate clinical collaboration and validation.

Potential applications include:

* Thyroid scintigraphy
* Parathyroid imaging
* Bone scintigraphy
* Hepatobiliary (HIDA) imaging
* Whole-body nuclear medicine imaging
* Other static scintigraphic studies

The ultimate goal is to develop clinically validated AI-assisted tools that can support nuclear medicine physicians across a broader range of static nuclear medicine examinations.

## 📚 Research Status

This work forms part of my **MPhil research in Physics**, with a focus on:

* Medical Physics
* Nuclear Medicine
* Medical Imaging
* Artificial Intelligence
* Deep Learning
* Explainable AI

**Manuscript submitted for publication.**

## ⚠️ Data Privacy and Responsible Use

This repository does not contain patient-identifiable information, clinical records, PACS credentials, or restricted hospital imaging data.

The repository is intended to document the **research methodology, software implementation, explainability approach, and shareable research materials**.

The application is a research prototype and is **not a medical device or standalone diagnostic system**. Clinical use would require appropriate external, prospective, and regulatory validation.

---

### 👩‍🔬 Researcher

**Marryum Zia**
MPhil Physics | Medical Physics | Nuclear Medicine | AI & Medical Imaging

