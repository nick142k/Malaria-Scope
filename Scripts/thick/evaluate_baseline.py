import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Malaria Scope")

TEST_CSV = PROJECT_ROOT / "reports" / "thick" / "test.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "thick_baseline"
    / "best_model.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "thick_baseline"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class ThickSmearDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.labels = (
            self.df["label_id"]
            .astype(int)
            .tolist()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = PROJECT_ROOT / row["image_path"]

        image = Image.open(image_path).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TEST TRANSFORM
# Must match validation transform used during training
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset = ThickSmearDataset(
    TEST_CSV,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("THICK SMEAR BASELINE - TEST EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Model: {MODEL_PATH}")


print("\nLoading ResNet-18...")

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Validation accuracy stored in checkpoint: "
    f"{checkpoint.get('val_accuracy', 'unknown')}"
)


# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        parasite_probabilities = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        predictions = (
            predictions
            .cpu()
            .numpy()
        )

        labels = (
            labels
            .cpu()
            .numpy()
        )

        all_labels.extend(labels.tolist())

        all_predictions.extend(
            predictions.tolist()
        )

        all_probabilities.extend(
            parasite_probabilities.tolist()
        )


# ============================================================
# NUMPY ARRAYS
# ============================================================

y_true = np.array(all_labels)

y_pred = np.array(all_predictions)

y_prob = np.array(all_probabilities)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

metrics = {

    "model": "ResNet-18",

    "checkpoint": str(
        MODEL_PATH
    ),

    "test_images": int(
        len(y_true)
    ),

    "class_definition": {
        "0": "parasite_absent",
        "1": "parasite_present"
    },

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall_sensitivity": float(
        recall
    ),

    "specificity": float(
        specificity
    ),

    "f1": float(
        f1
    ),

    "roc_auc": float(
        roc_auc
    ),

    "true_negative": int(tn),

    "false_positive": int(fp),

    "false_negative": int(fn),

    "true_positive": int(tp),

    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)]
    ]
}


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "test_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = test_dataset.df.copy()

predictions_df[
    "predicted_label_id"
] = y_pred

predictions_df[
    "parasite_probability"
] = y_prob

predictions_df[
    "correct"
] = (
    y_true == y_pred
)

predictions_df[
    "prediction"
] = np.where(
    y_pred == 1,
    "parasite_present",
    "parasite_absent"
)

predictions_path = (
    RESULTS_DIR
    / "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test images:       {len(y_true)}"
)

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall/Sensitivity: {recall:.4f}"
)

print(
    f"Specificity:        {specificity:.4f}"
)

print(
    f"F1 Score:           {f1:.4f}"
)

print(
    f"ROC-AUC:            {roc_auc:.4f}"
)

print("\nConfusion Matrix")
print(
    "                  Predicted"
)
print(
    "                 0       1"
)
print(
    f"Actual 0       {tn:4d}    {fp:4d}"
)
print(
    f"Actual 1       {fn:4d}    {tp:4d}"
)

print("\nClass interpretation:")
print("0 = parasite absent")
print("1 = parasite present")

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    f"Metrics:      {metrics_path}"
)

print(
    f"Predictions:  {predictions_path}"
)

print("=" * 60)
