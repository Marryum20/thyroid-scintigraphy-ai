# Explainability

This folder contains the code and documentation related to model explainability using Grad-CAM (Gradient-weighted Class Activation Mapping).

## Purpose

Grad-CAM was used to visualize the regions of thyroid scintigraphy images that contributed to the model's classification decision.

## Workflow

The explainability analysis involved:

1. Loading a trained deep learning model.
2. Providing a thyroid scintigraphy image as input.
3. Generating class-specific activation information.
4. Applying Grad-CAM to identify important image regions.
5. Creating visual heatmaps.
6. Overlaying the heatmaps on the original images for interpretation.

## Models

Explainability analysis can be applied to the trained deep learning architectures evaluated in this project, including:

* ConvNeXt
* DenseNet
* ResNet

## Purpose in Medical Imaging

The visual explanations provide an additional way to examine whether the model is focusing on clinically meaningful regions of the thyroid scintigraphy images.

## Privacy

Patient-identifiable images are not included in this repository.
