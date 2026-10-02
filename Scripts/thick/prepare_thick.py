
"""
prepare_thick.py
================

Prepare the baseline thick-smear classification dataset.

Task:
    Binary classification

Classes:
    infected   -> 1
    uninfected -> 0

Input:
    data/thick_smears/Thick_Infected/
    data/thick_smears/Thick_Uninfected/

Output:
    reports/thick/master_thick.csv
    reports/thick/invalid_thick_images.csv
    reports/thick/duplicate_thick_images.csv

IMPORTANT:
    - Original images are NOT moved.
    - Original images are NOT renamed.
    - Original images are NOT deleted.
    - No resizing or augmentation is performed here.
    - The 150-patient annotated dataset is intentionally excluded
      from this baseline preparation stage.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib

from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "thick"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET PATHS
# ============================================================

THICK_ROOT = DATA_DIR / "thick_smears"

INFECTED_DIR = THICK_ROOT / "Thick_Infected"
UNINFECTED_DIR = THICK_ROOT / "Thick_Uninfected"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# OUTPUT FILES
# ============================================================

MASTER_FILE = REPORT_DIR / "master_thick.csv"

INVALID_FILE = (
    REPORT_DIR
    / "invalid_thick_images.csv"
)

DUPLICATE_FILE = (
    REPORT_DIR
    / "duplicate_thick_images.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(directory):
    """Return all supported image files recursively."""

    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


def get_relative_path(path):
    """Return a project-relative path."""

    return str(
        path.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")


def calculate_md5(path):
    """
    Calculate MD5 hash.

    Used only for detecting exact duplicate files.
    """

    md5 = hashlib.md5()

    try:

        with open(
            path,
            "rb",
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:

        return ""


def inspect_image(path):
    """
    Validate an image and extract basic metadata.
    """

    try:

        with Image.open(path) as image:

            # Force Pillow to verify that the image can
            # actually be loaded.
            image.verify()

        # Re-open after verify() because verify() invalidates
        # the image object.
        with Image.open(path) as image:

            return {
                "valid": True,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "error": "",
            }

    except Exception as error:

        return {
            "valid": False,
            "width": "",
            "height": "",
            "mode": "",
            "format": "",
            "error": str(error),
        }


def get_file_size_mb(path):
    """Return file size in megabytes."""

    return round(
        path.stat().st_size / (1024 * 1024),
        4,
    )


# ============================================================
# PROCESS DATASET DIRECTORY
# ============================================================

def process_class(
    directory,
    label,
    numeric_label,
):
    """
    Process one classification directory.

    Parameters
    ----------
    directory : Path
        Dataset directory.

    label : str
        Human-readable class label.

    numeric_label : int
        Numeric ML label.
    """

    records = []
    invalid_records = []

    images = get_image_files(directory)

    print()
    print("-" * 70)
    print(f"CLASS       : {label}")
    print(f"LABEL       : {numeric_label}")
    print(f"DIRECTORY   : {directory}")
    print(f"IMAGE COUNT : {len(images):,}")
    print("-" * 70)

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        metadata = inspect_image(
            image_path
        )

        base_record = {
            "image_path": get_relative_path(
                image_path
            ),
            "filename": image_path.name,
            "label": label,
            "label_id": numeric_label,
            "smear_type": "thick",
            "source": "thick_baseline",
            "width": metadata["width"],
            "height": metadata["height"],
            "mode": metadata["mode"],
            "format": metadata["format"],
            "file_size_mb": get_file_size_mb(
                image_path
            ),
            "valid": metadata["valid"],
            "md5": "",
            "error": metadata["error"],
        }

        # ----------------------------------------------------
        # Invalid image
        # ----------------------------------------------------

        if not metadata["valid"]:

            invalid_records.append(
                base_record
            )

            continue

        # ----------------------------------------------------
        # Duplicate hash
        # ----------------------------------------------------

        base_record["md5"] = calculate_md5(
            image_path
        )

        records.append(
            base_record
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 500 == 0:

            print(
                f"Processed "
                f"{index:,}/{len(images):,}"
            )

    return records, invalid_records


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THICK SMEAR DATASET PREPARATION")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Dataset root : {THICK_ROOT}"
    )

    print(
        f"Report folder: {REPORT_DIR}"
    )

    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET DIRECTORY CHECK")
    print("=" * 70)

    print(
        f"Thick infected   : "
        f"{'FOUND' if INFECTED_DIR.exists() else 'MISSING'}"
    )

    print(
        f"Thick uninfected : "
        f"{'FOUND' if UNINFECTED_DIR.exists() else 'MISSING'}"
    )

    if not INFECTED_DIR.exists():

        raise FileNotFoundError(
            f"Missing directory:\n{INFECTED_DIR}"
        )

    if not UNINFECTED_DIR.exists():

        raise FileNotFoundError(
            f"Missing directory:\n{UNINFECTED_DIR}"
        )

    # ========================================================
    # PROCESS INFECTED
    # ========================================================

    infected_records, infected_invalid = process_class(
        directory=INFECTED_DIR,
        label="infected",
        numeric_label=1,
    )

    # ========================================================
    # PROCESS UNINFECTED
    # ========================================================

    uninfected_records, uninfected_invalid = process_class(
        directory=UNINFECTED_DIR,
        label="uninfected",
        numeric_label=0,
    )

    # ========================================================
    # COMBINE
    # ========================================================

    records = (
        infected_records
        + uninfected_records
    )

    invalid_records = (
        infected_invalid
        + uninfected_invalid
    )

    # ========================================================
    # DUPLICATE DETECTION
    # ========================================================

    print()
    print("=" * 70)
    print("DUPLICATE DETECTION")
    print("=" * 70)

    hash_to_paths = defaultdict(list)

    for record in records:

        file_hash = record["md5"]

        if file_hash:

            hash_to_paths[file_hash].append(
                record["image_path"]
            )

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths
        in hash_to_paths.items()
        if len(paths) > 1
    }

    duplicate_records = []

    for file_hash, paths in duplicate_groups.items():

        for path in paths:

            duplicate_records.append(
                {
                    "md5": file_hash,
                    "image_path": path,
                }
            )

    print(
        f"Duplicate groups : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images involved  : "
        f"{len(duplicate_records):,}"
    )

    # ========================================================
    # SAVE MASTER DATASET
    # ========================================================

    fieldnames = [
        "image_path",
        "filename",
        "label",
        "label_id",
        "smear_type",
        "source",
        "width",
        "height",
        "mode",
        "format",
        "file_size_mb",
        "valid",
        "md5",
        "error",
    ]

    with open(
        MASTER_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    # ========================================================
    # SAVE INVALID IMAGE REPORT
    # ========================================================

    with open(
        INVALID_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            invalid_records
        )

    # ========================================================
    # SAVE DUPLICATE REPORT
    # ========================================================

    with open(
        DUPLICATE_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "md5",
                "image_path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            duplicate_records
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("THICK DATASET SUMMARY")
    print("=" * 70)

    label_counts = Counter(
        record["label"]
        for record in records
    )

    for label in [
        "infected",
        "uninfected",
    ]:

        print(
            f"{label:15} : "
            f"{label_counts.get(label, 0):,}"
        )

    print()
    print(
        f"Total valid images   : "
        f"{len(records):,}"
    )

    print(
        f"Invalid images       : "
        f"{len(invalid_records):,}"
    )

    print(
        f"Duplicate groups     : "
        f"{len(duplicate_groups):,}"
    )

    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    dimension_counts = Counter(
        (
            record["width"],
            record["height"],
        )
        for record in records
    )

    print()
    print("Most common image dimensions:")

    for (
        (width, height),
        count,
    ) in dimension_counts.most_common(10):

        print(
            f"  {width} x {height}"
            f" : {count:,}"
        )

    # ========================================================
    # FORMATS
    # ========================================================

    format_counts = Counter(
        record["format"]
        for record in records
    )

    print()
    print("Image formats:")

    for image_format, count in (
        format_counts.most_common()
    ):

        print(
            f"  {str(image_format):10}"
            f" : {count:,}"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Master dataset : {MASTER_FILE}"
    )

    print(
        f"Invalid report : {INVALID_FILE}"
    )

    print(
        f"Duplicate report: {DUPLICATE_FILE}"
    )

    print()
    print("=" * 70)
    print("THICK DATASET PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
