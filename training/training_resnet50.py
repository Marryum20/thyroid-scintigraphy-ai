
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(img):
    img = np.array(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class ThyroidDataset(Dataset):

    def __init__(self, dataframe, class_to_idx, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["path"]
        label_name = row["label"]

        image = Image.open(image_path).convert("RGB")

        image = apply_clahe(image)

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[label_name]

        return image, label


# ============================================================
# MODEL
# ============================================================

class Model(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = timm.create_model( 
            "resnet50", 
            pretrained=True, 
            num_classes=num_classes 
            )

    def forward(self, x):

        return self.model(x)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = correct / total

    return val_loss, val_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = pd.read_csv("split_file.csv")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    classes = sorted(df["label"].unique())

    class_to_idx = {
        cls: i for i, cls in enumerate(classes)
    }

    num_classes = len(classes)

    print("Classes:", classes)
    print("Number of classes:", num_classes)
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = ThyroidDataset(
        train_df,
        class_to_idx,
        transform=train_transform
    )

    val_dataset = ThyroidDataset(
        val_df,
        class_to_idx,
        transform=val_transform
    )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Model(num_classes).to(device)


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_counts = train_df["label"].value_counts()

    class_weights = []

    for cls in classes:
        class_weights.append(
            1.0 / class_counts[cls]
        )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.02
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30
    )


    # --------------------------------------------------------
    # FREEZE BACKBONE
    # --------------------------------------------------------

    for name, param in model.named_parameters():

        if "stem" in name or "stages" in name:
            param.requires_grad = False


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    epochs = 35

    best_val_acc = 0.0

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []


    for epoch in range(epochs):

        # Unfreeze all layers after first epoch
        if epoch == 1:

            for param in model.parameters():
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "best_resnet50.pth"
            )

            print(
                f"Best model saved. "
                f"Validation accuracy: {best_val_acc:.4f}"
            )


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    plt.figure()

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Resnet50 Loss")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "resnet50_loss_curve.png",
        dpi=300
    )

    plt.close()


    plt.figure()

    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Resnet50 Accuracy")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "resnet50_accuracy_curve.png",
        dpi=300
    )

    plt.close()


    print("\nTraining completed.")
    print("Best validation accuracy:", best_val_acc)