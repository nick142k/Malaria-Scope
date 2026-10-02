"""
create_splits.py
================

Create reproducible train/validation/test splits for the
thick-smear binary classification dataset.

Input:
    reports/thick/master_thick.csv

Output:
    reports/thick/train.csv
    reports/thick/validation.csv
    reports/thick/test.csv

Split:
    70% Train
    15% Validation
    15% Test

The split is stratified by label so that infected and
uninfected samples remain proportionally represented.

IMPORTANT:
    - Original images are not moved.
    - Original images are not copied.
    - Original images are not modified.
    - The split is reproducible using a fixed random seed.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "thick"

INPUT_FILE = REPORT_DIR / "master_thick.csv"

TRAIN_FILE = REPORT_DIR / "train.csv"
VALIDATION_FILE = REPORT_DIR / "validation.csv"
TEST_FILE = REPORT_DIR / "test.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():
    """Validate split configuration."""

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test ratios "
            "must sum to 1.0."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """Load and validate the master dataset."""

    print()
    print("=" * 70)
    print("LOADING THICK DATASET")
    print("=" * 70)

    print(
        f"Input file : {INPUT_FILE}"
    )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "The master dataset is empty."
        )

    required_columns = {
        "image_path",
        "label",
        "label_id",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # Only valid images should enter the model splits.
    if "valid" in df.columns:

        valid_values = (
            df["valid"]
            .astype(str)
            .str.lower()
        )

        df = df[
            valid_values == "true"
        ].copy()

    # Remove rows with missing labels.
    df = df.dropna(
        subset=[
            "image_path",
            "label",
            "label_id",
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid samples remain after filtering."
        )

    # Remove exact duplicate image paths.
    df = df.drop_duplicates(
        subset=["image_path"]
    ).reset_index(drop=True)

    print(
        f"Usable images : {len(df):,}"
    )

    return df


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    df,
    name,
):
    """Print class distribution."""

    print()
    print(
        f"{name} class distribution:"
    )

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"  {label:12} : "
            f"{count:6,} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"  {'TOTAL':12} : "
        f"{len(df):6,}"
    )


# ============================================================
# CREATE SPLITS
# ============================================================

def create_splits(df):
    """
    Create stratified train/validation/test splits.

    First:
        70% train
        30% temporary

    Then:
        50% of temporary -> validation
        50% of temporary -> test

    Final result:
        70% train
        15% validation
        15% test
    """

    print()
    print("=" * 70)
    print("CREATING SPLITS")
    print("=" * 70)

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Second split:
    # Split temporary 50/50
    #
    # 30% temporary:
    #     15% validation
    #     15% test
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    # Reset indexes.
    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_df,
    validation_df,
    test_df,
):
    """Save split CSV files."""

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SPLITS SAVED")
    print("=" * 70)

    print(
        f"Train      : {TRAIN_FILE}"
    )

    print(
        f"Validation : {VALIDATION_FILE}"
    )

    print(
        f"Test       : {TEST_FILE}"
    )


# ============================================================
# VERIFY SPLITS
# ============================================================

def verify_splits(
    original_df,
    train_df,
    validation_df,
    test_df,
):
    """
    Verify split sizes, class distribution, and overlap.
    """

    print()
    print("=" * 70)
    print("VERIFYING SPLITS")
    print("=" * 70)

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Original samples : {len(original_df):,}"
    )

    print(
        f"Split samples    : {total:,}"
    )

    if total != len(original_df):

        raise RuntimeError(
            "Split sample count does not match "
            "the original dataset."
        )

    # --------------------------------------------------------
    # Check image-path overlap
    # --------------------------------------------------------

    train_paths = set(
        train_df["image_path"]
    )

    validation_paths = set(
        validation_df["image_path"]
    )

    test_paths = set(
        test_df["image_path"]
    )

    train_validation_overlap = (
        train_paths
        & validation_paths
    )

    train_test_overlap = (
        train_paths
        & test_paths
    )

    validation_test_overlap = (
        validation_paths
        & test_paths
    )

    print()
    print("Image overlap:")
    print(
        f"  Train/Validation : "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"  Train/Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation/Test  : "
        f"{len(validation_test_overlap)}"
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        raise RuntimeError(
            "Data leakage detected: "
            "the same image appears in multiple splits."
        )

    # --------------------------------------------------------
    # Print distributions
    # --------------------------------------------------------

    print_distribution(
        train_df,
        "TRAIN",
    )

    print_distribution(
        validation_df,
        "VALIDATION",
    )

    print_distribution(
        test_df,
        "TEST",
    )

    # --------------------------------------------------------
    # Print percentages
    # --------------------------------------------------------

    print()
    print("Overall split ratio:")

    print(
        f"  Train      : "
        f"{len(train_df) / total * 100:.2f}%"
    )

    print(
        f"  Validation : "
        f"{len(validation_df) / total * 100:.2f}%"
    )

    print(
        f"  Test       : "
        f"{len(test_df) / total * 100:.2f}%"
    )

    print()
    print(
        "No image-level overlap detected."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THICK SMEAR DATASET SPLITTING")
    print("=" * 70)

    print(
        f"Random seed : {RANDOM_SEED}"
    )

    print(
        f"Train       : {TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation  : {VALIDATION_RATIO:.0%}"
    )

    print(
        f"Test        : {TEST_RATIO:.0%}"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print_distribution(
        df,
        "FULL DATASET",
    )

    # --------------------------------------------------------
    # Create
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_splits(df)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_splits(
        df,
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_splits(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("THICK SMEAR SPLITTING COMPLETE")
    print("=" * 70)

    print()
    print("Next step:")
    print(
        "    python -u .\\Scripts\\thick\\train.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

