import os
import json
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ============================================================
# Configuration
# ============================================================

SEED = 42

# Change this if your split directory has a different name.
DATA_DIR = "./dataset"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = "./models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0  # Safe for Windows


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Data
# ============================================================

def create_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found:\n{TRAIN_DIR}\n\n"
            "Check DATA_DIR and your split directory structure."
        )

    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(
            f"Validation directory not found:\n{VAL_DIR}\n\n"
            "Check DATA_DIR and your split directory structure."
        )

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("\nDataset information")
    print("-" * 50)

    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")

    print(f"\nClasses:")
    print(train_dataset.classes)

    print(f"\nClass mapping:")
    print(train_dataset.class_to_idx)

    return train_loader, val_loader, train_dataset, val_dataset


# ============================================================
# Model
# ============================================================

def create_model(num_classes):

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Freeze the feature extractor initially.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final classification layer.
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )

    return model


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

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

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()

    print("=" * 60)
    print("Malaria Blood Smear Classification")
    print("=" * 60)

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, train_dataset, val_dataset = \
        create_dataloaders()

    # Make sure both datasets use the same classes.
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation classes do not match.\n"
            f"Train: {train_dataset.classes}\n"
            f"Val:   {val_dataset.classes}"
        )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(num_classes)

    model = model.to(device)

    print("\nModel: ResNet-18")
    print(f"Number of classes: {num_classes}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # Only train parameters that require gradients.
    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"| "
            f"Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| "
            f"Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f} "
            f"| LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_dataset.class_to_idx,
                "image_size": IMAGE_SIZE,
                "model_name": "resnet18"
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print(
                f"  -> Best model saved "
                f"(val_loss={val_loss:.4f})"
            )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.load_state_dict(best_model_weights)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "model_name": "resnet18"
    }

    torch.save(
        final_checkpoint,
        LAST_MODEL_PATH
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    print(
        f"Best model:\n"
        f"  {BEST_MODEL_PATH}"
    )

    print(
        f"\nLast model:\n"
        f"  {LAST_MODEL_PATH}"
    )

    print(
        f"\nTraining history:\n"
        f"  {HISTORY_PATH}"
    )

    print("\nClasses:")
    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")


if __name__ == "__main__":
    main()
