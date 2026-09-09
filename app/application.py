
"""
Thyroid Scintigraphy AI Demonstration App

A Streamlit-based research demonstration for deep learning classification
of thyroid scintigraphy images.

Supported architectures:
    - ConvNeXt Tiny
    - ConvNeXt Small
    - DenseNet-201
    - ResNet-50

Input:
    DICOM thyroid scintigraphy image

Processing:
    DICOM pixel data
    -> intensity normalization
    -> CLAHE
    -> RGB conversion
    -> 320 x 320 resizing
    -> ImageNet normalization

Output:
    - Predicted class
    - Prediction confidence
    - Class probabilities
    - Grad-CAM visualization

This application is intended for research and demonstration purposes only.
It is not a medical device and should not be used as a standalone diagnostic
system.
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import pydicom
import streamlit as st
import timm
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

# Import Grad-CAM function from the explainability module
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from explainability.gradcam import generate_gradcam


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thyroid Scintigraphy AI",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 320
NUM_CLASSES = 5

CLASSES = [
    "Diffuse Goiter",
    "Hyperfunctioning",
    "Hypofunctioning",
    "MNG",
    "Normal",
]


CLASS_DESCRIPTIONS = {
    "Diffuse Goiter": (
        "The model classified the scintigraphic pattern as consistent "
        "with diffuse thyroid enlargement."
    ),

    "Hyperfunctioning": (
        "The model classified the scintigraphic pattern as consistent "
        "with increased focal or diffuse functional activity."
    ),

    "Hypofunctioning": (
        "The model classified the scintigraphic pattern as consistent "
        "with reduced functional activity."
    ),

    "MNG": (
        "The model classified the scintigraphic pattern as consistent "
        "with a multinodular thyroid uptake pattern."
    ),

    "Normal": (
        "The model classified the scintigraphic pattern as consistent "
        "with a normal thyroid scintigraphy pattern."
    ),
}


# ============================================================
# MODEL PATHS
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# Replace these paths with your local model-weight locations.
#
# Do NOT upload private model weights to GitHub unless you have
# permission to distribute them.
# ------------------------------------------------------------

MODEL_PATHS = {
    "ConvNeXt Tiny": os.path.join(
        PROJECT_ROOT,
        "models",
        "convnext_tiny_320.pth",
    ),

    "ConvNeXt Small": os.path.join(
        PROJECT_ROOT,
        "models",
        "convnext_small_320.pth",
    ),

    "DenseNet-201": os.path.join(
        PROJECT_ROOT,
        "models",
        "densenet201_320.pth",
    ),

    "ResNet-50": os.path.join(
        PROJECT_ROOT,
        "models",
        "resnet50_320.pth",
    ),
}


MODEL_ARCHITECTURES = {
    "ConvNeXt Tiny": "convnext_tiny",
    "ConvNeXt Small": "convnext_small",
    "DenseNet-201": "densenet201",
    "ResNet-50": "resnet50",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

class ThyroidClassifier(nn.Module):
    """
    Wrapper around the timm classification models.

    The backbone attribute is intentionally retained so that the
    model structure is consistent with the Grad-CAM implementation.
    """

    def __init__(self, architecture, num_classes=NUM_CLASSES):
        super().__init__()

        self.backbone = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_name):
    """
    Load a trained model and cache it for the Streamlit session.
    """

    architecture = MODEL_ARCHITECTURES[model_name]
    model_path = MODEL_PATHS[model_name]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights were not found:\n{model_path}\n\n"
            "Please place the trained model weights in the configured "
            "models directory."
        )

    model = ThyroidClassifier(
        architecture=architecture,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    # Handle common checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def apply_clahe(image):
    """
    Apply CLAHE contrast enhancement.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image.

    Returns
    -------
    numpy.ndarray
        CLAHE-enhanced grayscale image.
    """

    image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(image)


def normalize_image(image):
    """
    Min-max normalize an image to 8-bit intensity.
    """

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    image = (
        (image - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return image.astype(np.uint8)


def extract_dicom_image(ds):
    """
    Extract a usable 2D image from a DICOM dataset.

    For multi-frame data, the first frame is used in this public
    demonstration version.

    If the research preprocessing uses a different projection
    strategy for multi-frame DICOM data, that strategy should be
    matched before using this application for research evaluation.
    """

    image = ds.pixel_array

    if image.ndim == 3:
        image = image[0]

    image = normalize_image(image)

    image = apply_clahe(image)

    return image


def prepare_display_image(image):
    """
    Convert a grayscale numpy array to an RGB PIL image.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB,
    )

    return Image.fromarray(rgb)


# ============================================================
# MODEL TRANSFORM
# ============================================================

MODEL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image):
    """
    Run model inference.

    Returns
    -------
    predicted_class : str
    confidence : float
    probabilities : numpy.ndarray
    """

    input_tensor = MODEL_TRANSFORM(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    probabilities = probabilities[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASSES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities,
    )


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("🔬 Thyroid Scintigraphy AI")

st.markdown(
    """
### AI-assisted classification of thyroid scintigraphy images

This research demonstration uses deep learning models to classify
thyroid scintigraphy patterns into five categories.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select AI Architecture",
    list(MODEL_ARCHITECTURES.keys()),
)

st.sidebar.markdown("---")

st.sidebar.subheader("Supported Classes")

for class_name in CLASSES:
    st.sidebar.write(f"• {class_name}")

st.sidebar.markdown("---")

st.sidebar.info(
    f"Processing resolution: {IMAGE_SIZE} × {IMAGE_SIZE}\n\n"
    f"Device: {DEVICE}"
)


# ============================================================
# UPLOAD
# ============================================================

st.subheader("Upload Thyroid Scintigraphy DICOM")

uploaded_file = st.file_uploader(
    "Select a DICOM (.dcm) file",
    type=["dcm", "dicom"],
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        ds = pydicom.dcmread(
            uploaded_file,
            force=True,
        )

        raw_image = extract_dicom_image(ds)

        display_image = prepare_display_image(
            raw_image
        )

        st.subheader("Input Image")

        st.image(
            display_image,
            caption="Processed thyroid scintigraphy image",
            use_container_width=True,
        )

        st.markdown("---")

        analyze_button = st.button(
            "🔍 Analyze Scan",
            type="primary",
        )


        # ====================================================
        # ANALYSIS
        # ====================================================

        if analyze_button:

            try:

                start_time = time.time()

                with st.spinner(
                    f"Loading {selected_model}..."
                ):

                    model = load_model(
                        selected_model
                    )


                with st.spinner(
                    "Analyzing scintigraphy image..."
                ):

                    prediction, confidence, probabilities = predict(
                        model,
                        display_image,
                    )

                    architecture = MODEL_ARCHITECTURES[
                        selected_model
                    ]

                    gradcam_image = generate_gradcam(
                        model=model,
                        image=display_image,
                        architecture=architecture,
                        transform=MODEL_TRANSFORM,
                        device=DEVICE,
                    )

                elapsed_time = time.time() - start_time


                # ==========================================
                # RESULT
                # ==========================================

                st.success(
                    f"Prediction: {prediction}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Predicted Class",
                        prediction,
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%",
                    )

                with col3:

                    st.metric(
                        "Inference Time",
                        f"{elapsed_time:.2f} s",
                    )


                # ==========================================
                # PROBABILITIES
                # ==========================================

                st.subheader(
                    "Class Probability"
                )

                probability_df = pd.DataFrame(
                    {
                        "Class": CLASSES,
                        "Probability (%)": (
                            probabilities * 100
                        ),
                    }
                )

                probability_df = probability_df.sort_values(
                    "Probability (%)",
                    ascending=False,
                )

                st.bar_chart(
                    probability_df.set_index(
                        "Class"
                    )
                )


                # ==========================================
                # GRAD-CAM
                # ==========================================

                st.subheader(
                    "Grad-CAM Explainability"
                )

                st.write(
                    """
                    The Grad-CAM visualization highlights regions of the
                    scintigraphy image that contributed to the model's
                    prediction. It is provided for model interpretability
                    and should not be considered an independent diagnostic
                    result.
                    """
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        display_image,
                        caption="Processed input",
                        use_container_width=True,
                    )

                with col2:

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM visualization",
                        use_container_width=True,
                    )


                # ==========================================
                # CLINICAL-STYLE SUMMARY
                # ==========================================

                st.subheader(
                    "AI Classification Summary"
                )

                st.info(
                    CLASS_DESCRIPTIONS[prediction]
                )

                st.write(
                    f"""
                    **Model:** {selected_model}

                    **Predicted class:** {prediction}

                    **Model confidence:** {confidence * 100:.2f}%

                    **Input resolution:** {IMAGE_SIZE} × {IMAGE_SIZE}
                    """
                )


                # ==========================================
                # DISCLAIMER
                # ==========================================

                st.warning(
                    """
                    **Research Demonstration Disclaimer**

                    This application is intended for research,
                    educational, and demonstration purposes only.

                    The AI output should not be used as a standalone
                    clinical diagnosis or as a replacement for assessment
                    by a qualified nuclear medicine physician.

                    Further clinical validation, external validation,
                    and prospective evaluation are required before
                    clinical deployment.
                    """
                )


            except FileNotFoundError as error:

                st.error(
                    str(error)
                )

                st.info(
                    """
                    Add the trained model weights to the configured
                    `models/` directory or update `MODEL_PATHS` in
                    `app.py`.
                    """
                )


            except RuntimeError as error:

                st.error(
                    "The model weights could not be loaded."
                )

                st.exception(error)

                st.info(
                    """
                    Check that the model architecture and checkpoint
                    structure match the model used during training.
                    """
                )


            except Exception as error:

                st.error(
                    "An error occurred during analysis."
                )

                st.exception(error)


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

with st.expander("About this project"):

    st.markdown(
        """
        ### Research Project

        This application demonstrates an AI-based framework for the
        classification of thyroid scintigraphy images.

        The research framework evaluates four deep learning architectures:

        - ConvNeXt Tiny
        - ConvNeXt Small
        - DenseNet-201
        - ResNet-50

        The classification task contains five scintigraphic categories:

        - Diffuse Goiter
        - Hyperfunctioning
        - Hypofunctioning
        - Multinodular Goiter (MNG)
        - Normal

        The application includes Grad-CAM explainability to provide
        visual insight into regions influencing the model prediction.

        ### Intended Role

        The proposed system is designed as an AI-assisted tool that may
        support nuclear medicine physicians in interpreting thyroid
        scintigraphy images.

        It is not intended to replace expert clinical judgment.
        """
    )


st.caption(
    "Thyroid Scintigraphy AI | Research Demonstration"
)

