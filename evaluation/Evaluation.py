
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# SETTINGS
# ============================================================

# CSV containing image paths, classes and dataset split
CSV_FILE = "split_file.csv"

# Select the model to evaluate
MODEL_NAME = "convnext_small"

# Path to the trained model weights
MODEL_WEIGHTS = "best_convnext_small.pth"

# Evaluation split: "val" or "test"
EVALUATION_SPLIT = "test"

# Output directory
OUTPUT_DIR = "test_results"

# Image size used during model training
IMAGE_SIZE = 320

# Batch size
BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    out = cv2.merge(
        (cl, a, b)
    )

    out = cv2.cvtColor(
        out,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(out)


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class ThyroidDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_idx
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.class_to_idx = class_to_idx

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply the same CLAHE preprocessing
        # used during model development
        image = apply_clahe(image)

        image = val_transform(image)

        label_name = row["class"]

        label = torch.tensor(
            self.class_to_idx[label_name],
            dtype=torch.long
        )

        return (
            image,
            label,
            image_path
        )


# ============================================================
# MODEL
# ============================================================

class Model(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes
    ):

        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_name,
    model_weights,
    num_classes
):

    model = Model(
        model_name,
        num_classes
    )

    checkpoint = torch.load(
        model_weights,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    model.eval()

    print(
        f"\nLoaded model: {model_name}"
    )

    print(
        f"Weights: {model_weights}"
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    classes,
    output_dir,
    phase="test",
    save_misclassified=True
):

    model.eval()

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    predictions = []
    targets = []
    all_probabilities = []

    misclassified = []

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        for images, labels, paths in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            # ------------------------------------------------
            # MISCLASSIFIED IMAGES
            # ------------------------------------------------

            for i in range(len(paths)):

                true_label = labels[i].item()

                predicted_label = predicted[i].item()

                if true_label != predicted_label:

                    true_class = classes[
                        true_label
                    ]

                    predicted_class = classes[
                        predicted_label
                    ]

                    confidence_value = (
                        confidence[i].item()
                    )

                    source_path = paths[i]

                    misclassified.append({

                        "path": source_path,

                        "true_class":
                            true_class,

                        "predicted_class":
                            predicted_class,

                        "confidence":
                            confidence_value
                    })

                    if save_misclassified:

                        save_dir = os.path.join(

                            output_dir,

                            "misclassified",

                            f"true_{true_class}",

                            f"predicted_{predicted_class}"
                        )

                        os.makedirs(
                            save_dir,
                            exist_ok=True
                        )

                        filename = os.path.basename(
                            source_path
                        )

                        destination = os.path.join(
                            save_dir,
                            filename
                        )

                        shutil.copy(
                            source_path,
                            destination
                        )


    predictions = np.array(
        predictions
    )

    targets = np.array(
        targets
    )

    all_probabilities = np.array(
        all_probabilities
    )


    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        targets,
        predictions
    )

    print("\n===================================")
    print(f"{phase.upper()} EVALUATION")
    print("===================================")

    print(
        f"\nOverall Accuracy: "
        f"{accuracy:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(

        targets,

        predictions,

        target_names=classes,

        zero_division=0,

        output_dict=True
    )

    report_text = classification_report(

        targets,

        predictions,

        target_names=classes,

        zero_division=0
    )


    print("\n===================================")
    print("CLASSIFICATION REPORT")
    print("===================================")

    print(report_text)


    report_file = os.path.join(

        output_dir,

        f"{phase}_classification_report.txt"
    )

    with open(
        report_file,
        "w"
    ) as file:

        file.write(
            report_text
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        targets,

        predictions,

        labels=np.arange(
            len(classes)
        )
    )


    print("\n===================================")
    print("CONFUSION MATRIX")
    print("===================================")

    print(cm)


    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=classes,

        yticklabels=classes
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        f"{phase.capitalize()} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_dir,

            f"{phase}_confusion_matrix.png"
        ),

        dpi=300
    )

    plt.close()


    # ========================================================
    # CLINICAL METRICS
    # ========================================================

    metrics_data = []


    print("\n===================================")
    print("CLASS-WISE METRICS")
    print("===================================")


    for i, class_name in enumerate(classes):

        tp = cm[i, i]

        fn = np.sum(
            cm[i, :]
        ) - tp

        fp = np.sum(
            cm[:, i]
        ) - tp

        tn = np.sum(cm) - (
            tp + fn + fp
        )


        precision = tp / (
            tp + fp + 1e-8
        )

        sensitivity = tp / (
            tp + fn + 1e-8
        )

        specificity = tn / (
            tn + fp + 1e-8
        )

        npv = tn / (
            tn + fn + 1e-8
        )

        f1 = (
            2 * precision * sensitivity
            /
            (
                precision
                + sensitivity
                + 1e-8
            )
        )

        class_accuracy = (
            (tp + tn)
            /
            (
                tp
                + tn
                + fp
                + fn
                + 1e-8
            )
        )


        metrics_data.append({

            "Class":
                class_name,

            "Precision":
                precision,

            "Recall":
                sensitivity,

            "Sensitivity":
                sensitivity,

            "Specificity":
                specificity,

            "PPV":
                precision,

            "NPV":
                npv,

            "F1-Score":
                f1,

            "Accuracy":
                class_accuracy
        })


        print(
            f"\nClass: {class_name}"
        )

        print(
            f"Precision   : "
            f"{precision:.4f}"
        )

        print(
            f"Sensitivity : "
            f"{sensitivity:.4f}"
        )

        print(
            f"Specificity : "
            f"{specificity:.4f}"
        )

        print(
            f"PPV         : "
            f"{precision:.4f}"
        )

        print(
            f"NPV         : "
            f"{npv:.4f}"
        )

        print(
            f"F1-Score    : "
            f"{f1:.4f}"
        )

        print(
            f"Accuracy    : "
            f"{class_accuracy:.4f}"
        )


    metrics_df = pd.DataFrame(
        metrics_data
    )


    # ========================================================
    # MACRO-AVERAGED METRICS
    # ========================================================

    macro_precision = (
        metrics_df["Precision"].mean()
    )

    macro_sensitivity = (
        metrics_df["Sensitivity"].mean()
    )

    macro_specificity = (
        metrics_df["Specificity"].mean()
    )

    macro_ppv = (
        metrics_df["PPV"].mean()
    )

    macro_npv = (
        metrics_df["NPV"].mean()
    )

    macro_f1 = (
        metrics_df["F1-Score"].mean()
    )

    macro_accuracy = (
        metrics_df["Accuracy"].mean()
    )


    print("\n===================================")
    print("OVERALL METRICS")
    print("===================================")

    print(
        f"Overall Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean Precision   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Mean Sensitivity : "
        f"{macro_sensitivity:.4f}"
    )

    print(
        f"Mean Specificity : "
        f"{macro_specificity:.4f}"
    )

    print(
        f"Mean PPV         : "
        f"{macro_ppv:.4f}"
    )

    print(
        f"Mean NPV         : "
        f"{macro_npv:.4f}"
    )

    print(
        f"Mean F1-Score    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Mean Accuracy    : "
        f"{macro_accuracy:.4f}"
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_file = os.path.join(

        output_dir,

        f"{phase}_metrics.csv"
    )

    metrics_df.to_csv(

        metrics_file,

        index=False
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    y_binary = label_binarize(

        targets,

        classes=np.arange(
            len(classes)
        )
    )


    roc_auc_values = []


    plt.figure(
        figsize=(8, 6)
    )


    for i, class_name in enumerate(classes):

        fpr, tpr, _ = roc_curve(

            y_binary[:, i],

            all_probabilities[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        roc_auc_values.append(
            roc_auc
        )


        plt.plot(

            fpr,

            tpr,

            label=(
                f"{class_name} "
                f"AUC={roc_auc:.3f}"
            )
        )


    plt.plot(

        [0, 1],

        [0, 1],

        "k--"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{phase.capitalize()} ROC Curve"
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            output_dir,

            f"{phase}_roc_curve.png"
        ),

        dpi=300
    )

    plt.close()


    macro_auc = np.mean(
        roc_auc_values
    )


    print(
        f"\nMacro ROC-AUC: "
        f"{macro_auc:.4f}"
    )


    # ========================================================
    # SAVE ROC-AUC
    # ========================================================

    auc_df = pd.DataFrame({

        "Class": classes,

        "ROC_AUC": roc_auc_values
    })


    auc_df.loc[
        len(auc_df)
    ] = [
        "Macro Average",
        macro_auc
    ]


    auc_df.to_csv(

        os.path.join(

            output_dir,

            f"{phase}_roc_auc.csv"
        ),

        index=False
    )


    # ========================================================
    # SAVE MISCLASSIFIED CSV
    # ========================================================

    if save_misclassified:

        misclassified_df = pd.DataFrame(
            misclassified
        )

        misclassified_file = os.path.join(

            output_dir,

            "misclassified.csv"
        )

        misclassified_df.to_csv(

            misclassified_file,

            index=False
        )

        print(
            f"\nMisclassified images: "
            f"{len(misclassified)}"
        )

        print(
            f"Saved CSV: "
            f"{misclassified_file}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "Model":
            MODEL_NAME,

        "Split":
            phase,

        "Accuracy":
            accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Sensitivity":
            macro_sensitivity,

        "Macro Specificity":
            macro_specificity,

        "Macro PPV":
            macro_ppv,

        "Macro NPV":
            macro_npv,

        "Macro F1":
            macro_f1,

        "Macro ROC-AUC":
            macro_auc
    }


    summary_df = pd.DataFrame(
        [summary]
    )


    summary_df.to_csv(

        os.path.join(

            output_dir,

            f"{phase}_summary.csv"
        ),

        index=False
    )


    print(
        "\nEvaluation completed successfully."
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_FILE
    )


    # --------------------------------------------------------
    # GET CLASSES
    # --------------------------------------------------------

    classes = sorted(
        df["class"].unique()
    )


    class_to_idx = {

        class_name: index

        for index, class_name
        in enumerate(classes)
    }


    print(
        "\nClasses:"
    )

    print(
        class_to_idx
    )


    # --------------------------------------------------------
    # SELECT EVALUATION DATA
    # --------------------------------------------------------

    evaluation_df = df[
        df["split"] ==
        EVALUATION_SPLIT
    ].copy()


    print(
        f"\nEvaluation split: "
        f"{EVALUATION_SPLIT}"
    )

    print(
        f"Number of images: "
        f"{len(evaluation_df)}"
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    evaluation_dataset = ThyroidDataset(

        evaluation_df,

        class_to_idx
    )


    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    evaluation_loader = DataLoader(

        evaluation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(

        MODEL_NAME,

        MODEL_WEIGHTS,

        len(classes)
    )


    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    evaluate_model(

        model,

        evaluation_loader,

        classes,

        OUTPUT_DIR,

        phase=EVALUATION_SPLIT,

        save_misclassified=True
    )
