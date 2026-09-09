
"""
Grad-CAM Explainability for Thyroid Scintigraphy Classification

This module provides Grad-CAM visualization for the four CNN architectures
used in the thyroid scintigraphy classification project:

- ConvNeXt Tiny
- ConvNeXt Small
- DenseNet-201
- ResNet-50

Grad-CAM highlights image regions that contribute most strongly to the
model's predicted class.

Note:
Grad-CAM is intended for model interpretability and visualization.
It should not be considered an independent diagnostic method.
"""

import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model, architecture):
    """
    Select an appropriate target layer for Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded classification model.

    architecture : str
        Model architecture name.

    Returns
    -------
    torch.nn.Module
        Target layer used for Grad-CAM.
    """

    architecture = architecture.lower()

    if architecture == "convnext_tiny":
        return model.backbone.stages[-1].blocks[-1]

    elif architecture == "convnext_small":
        return model.backbone.stages[-1].blocks[-2]

    elif architecture == "densenet201":
        return model.backbone.features[-2]

    elif architecture == "resnet50":
        return model.backbone.layer4[-1]

    else:
        raise ValueError(
            f"Unsupported architecture for Grad-CAM: {architecture}"
        )


def generate_gradcam(
    model,
    image,
    architecture,
    transform,
    device="cpu",
    image_weight=0.57,
):
    """
    Generate a Grad-CAM visualization.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.

    image : PIL.Image.Image
        Input thyroid scintigraphy image.

    architecture : str
        Model architecture name.

    transform : torchvision transform
        Preprocessing transformation used by the model.

    device : str
        Inference device.

    image_weight : float
        Weight of the original image in the visualization.

    Returns
    -------
    numpy.ndarray
        RGB Grad-CAM visualization.
    """

    model.eval()

    # Convert image to RGB
    rgb_image = image.convert("RGB")

    # Normalize image for visualization
    rgb_array = np.asarray(rgb_image).astype(np.float32) / 255.0

    # Prepare model input
    input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    # Select target layer
    target_layer = get_target_layer(model, architecture)

    # Generate Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Overlay CAM on original image
    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True,
        image_weight=image_weight,
    )

    return visualization
