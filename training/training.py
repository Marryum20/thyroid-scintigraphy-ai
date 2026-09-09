import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

# ============================================================

# DEVICE

# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================

# CLAHE

# ============================================================

def apply_clahe(image):

```
img = np.array(image)

if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

cl = clahe.apply(l)

out = cv2.merge((cl, a, b))

out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

return Image.fromarray(out)
```

# ============================================================

# TRANSFORMS

# ============================================================

train_transform = transforms.Compose([
transforms.Resize((320, 320)),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10),
transforms.ToTensor(),
transforms.Normalize(
[0.485, 0.456, 0.406],
[0.229, 0.224, 0.225]
)
])

val_transform = transforms.Compose([
transforms.Resize((320, 320)),
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

```
def __init__(self, dataframe, class_to_idx, train=True):

    self.df = dataframe.reset_index(drop=True)
    self.class_to_idx = class_to_idx
    self.train = train

def __len__(self):

    return len(self.df)

def __getitem__(self, idx):

    row = self.df.iloc[idx]

    img = Image.open(row["path"]).convert("RGB")

    # CLAHE enhancement
    img = apply_clahe(img)

    label_name = row["class"]

    if self.train:
        img = train_transform(img)
    else:
        img = val_transform(img)

    label = torch.tensor(
        self.class_to_idx[label_name],
        dtype=torch.long
    )

    return img, label
```

# ============================================================

# MODEL

# ============================================================

class Model(nn.Module):

```
def __init__(self, num_classes):

    super().__init__()

    self.backbone = timm.create_model(
        "convnext_small",
        pretrained=True,
        num_classes=num_classes
    )

def forward(self, x):

    return self.backbone(x)
```

# ============================================================

# TRAIN ONE EPOCH

# ============================================================

def train_one_epoch(
model,
loader,
optimizer,
criterion
):

```
model.train()

total_loss = 0
correct = 0
total = 0

for x, y in loader:

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

    predictions = torch.argmax(output, dim=1)

    correct += (predictions == y).sum().item()

    total += y.size(0)

accuracy = correct / total

average_loss = total_loss / len(loader)

return average_loss, accuracy
```

# ============================================================

# VALIDATION

# ============================================================

def validate(
model,
loader,
criterion
):

```
model.eval()

total_loss = 0
correct = 0
total = 0

with torch.no_grad():

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        loss = criterion(output, y)

        total_loss += loss.item()

        predictions = torch.argmax(output, dim=1)

        correct += (predictions == y).sum().item()

        total += y.size(0)

accuracy = correct / total

average_loss = total_loss / len(loader)

return average_loss, accuracy
```

# ============================================================

# MAIN

# ============================================================

if **name** == "**main**":

```
# ========================================================
# LOAD DATA SPLIT
# ========================================================

df = pd.read_csv("split_file.csv")

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

classes = sorted(df["class"].unique())

class_to_idx = {
    c: i for i, c in enumerate(classes)
}

print("\nClasses:")
print(class_to_idx)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))


# ========================================================
# DATASETS
# ========================================================

train_ds = ThyroidDataset(
    train_df,
    class_to_idx,
    train=True
)

val_ds = ThyroidDataset(
    val_df,
    class_to_idx,
    train=False
)


# ========================================================
# DATALOADERS
# ========================================================

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False
)


# ========================================================
# MODEL
# ========================================================

model = Model(len(classes)).to(device)


# ========================================================
# CLASS WEIGHTS
# ========================================================

class_counts = train_df["class"].value_counts()

weights = []

for c in classes:

    weights.append(
        1.0 / class_counts[c]
    )

class_weights = torch.tensor(
    weights,
    dtype=torch.float
).to(device)


# ========================================================
# LOSS FUNCTION
# ========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.02
)


# ========================================================
# OPTIMIZER
# ========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ========================================================
# LEARNING RATE SCHEDULER
# ========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30
)


# ========================================================
# FREEZE BACKBONE
# ========================================================

for param in model.backbone.stem.parameters():

    param.requires_grad = False

for param in model.backbone.stages.parameters():

    param.requires_grad = False


# ========================================================
# TRAINING HISTORY
# ========================================================

best_val_accuracy = 0

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# ========================================================
# TRAINING LOOP
# ========================================================

for epoch in range(35):

    print(
        f"\n================ EPOCH {epoch + 1} ================"
    )


    # ----------------------------------------------------
    # UNFREEZE BACKBONE AFTER FIRST EPOCH
    # ----------------------------------------------------

    if epoch == 1:

        for param in model.backbone.parameters():

            param.requires_grad = True

        print("Backbone unfrozen")


    # ----------------------------------------------------
    # TRAIN
    # ----------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    # ----------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )


    # ----------------------------------------------------
    # UPDATE LEARNING RATE
    # ----------------------------------------------------

    scheduler.step()


    # ----------------------------------------------------
    # SAVE HISTORY
    # ----------------------------------------------------

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    # ----------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------

    print(f"\nEpoch: {epoch + 1}")

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # ----------------------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_convnext_small.pth"
        )

        print("Best model saved.")


# ========================================================
# TRAINING CURVES
# ========================================================

epochs = range(
    1,
    len(train_losses) + 1
)


# --------------------------------------------------------
# LOSS CURVE
# --------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.title(
    "ConvNeXt Small Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "convnext_small_loss_curve.png"
)

plt.close()


# --------------------------------------------------------
# ACCURACY CURVE
# --------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.title(
    "ConvNeXt Small Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "convnext_small_accuracy_curve.png"
)

plt.close()


print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Best model: best_convnext_small.pth"
)

print("Training curves saved.")
```
