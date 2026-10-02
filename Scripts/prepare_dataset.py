
"""
prepare_dataset.py
==================

Creates a master dataset index from the existing malaria blood-smear dataset.

IMPORTANT:
- Original images are NOT moved.
- Original images are NOT renamed.
- Original images are NOT deleted.
- This script only reads the dataset and creates CSV metadata.

Output:
    reports/master_dataset.csv

Columns:
    image_path
    filename
    smear_type
    label
    patient_id
    source
    width
    height
    mode
    format
    file_size_mb
"""

from pathlib import Path
import csv
import hashlib
from collections import Counter

from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET PATHS
# ============================================================

THICK_ROOT = DATA_DIR / "thick_smears"
THIN_ROOT = DATA_DIR / "thin_smear"

THICK_INFECTED = THICK_ROOT / "Thick_Infected"
THICK_UNINFECTED = THICK_ROOT / "Thick_Uninfected"

THIN_PARASITIZED = THIN_ROOT / "Parasitized"
THIN_UNINFECTED = THIN_ROOT / "Uninfected"

THICK_PATIENT_ROOT = (
    THICK_ROOT
    / "150_patient_thick_annoted_blood_smears"
)

GT_ROOT = THICK_PATIENT_ROOT / "GT_updated"


# ============================================================
# IMAGE EXTENSIONS
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
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )


def get_image_metadata(image_path):
    """
    Read basic image metadata.

    Returns:
        width, height, mode, format
    """

    try:

        with Image.open(image_path) as image:

            return {
                "valid": True,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }

    except Exception:

        return {
            "valid": False,
            "width": "",
            "height": "",
            "mode": "",
            "format": "",
        }


def calculate_md5(image_path):
    """Calculate MD5 hash for duplicate detection."""

    md5 = hashlib.md5()

    try:

        with open(image_path, "rb") as file:

            while chunk := file.read(1024 * 1024):

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:

        return ""


def get_relative_path(path):
    """Convert absolute path to project-relative path."""

    return str(
        path.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")


def get_file_size_mb(path):
    """Return file size in MB."""

    return round(
        path.stat().st_size / (1024 * 1024),
        4,
    )


def find_patient_id(image_path):
    """
    Attempt to identify patient/sample ID.

    For images inside:

        150_patient_thick_annoted_blood_smears/
            GT_updated/
                TF100_CS39/
                TF101_233/
                ...

    the immediate directory below GT_updated is treated
    as the patient/sample ID.

    For classification folders where patient information
    is unavailable, 'unknown' is returned.
    """

    try:

        relative = image_path.relative_to(GT_ROOT)

        parts = relative.parts

        if len(parts) >= 2:

            return parts[0]

    except ValueError:
        pass

    return "unknown"


# ============================================================
# DATASET RECORD CREATION
# ============================================================

records = []


def process_directory(
    directory,
    smear_type,
    label,
    source,
):
    """
    Process all images from a dataset directory.
    """

    images = get_image_files(directory)

    print()
    print("-" * 70)
    print(f"Processing: {source}")
    print(f"Directory : {directory}")
    print(f"Images    : {len(images):,}")
    print("-" * 70)

    for index, image_path in enumerate(images, start=1):

        metadata = get_image_metadata(image_path)

        patient_id = find_patient_id(image_path)

        file_hash = calculate_md5(image_path)

        record = {
            "image_path": get_relative_path(image_path),
            "filename": image_path.name,
            "smear_type": smear_type,
            "label": label,
            "patient_id": patient_id,
            "source": source,
            "width": metadata["width"],
            "height": metadata["height"],
            "mode": metadata["mode"],
            "format": metadata["format"],
            "file_size_mb": get_file_size_mb(image_path),
            "valid": metadata["valid"],
            "md5": file_hash,
        }

        records.append(record)

        if index % 1000 == 0:

            print(
                f"  Processed {index:,}/{len(images):,}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MALARIA DATASET PREPARATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data folder  : {DATA_DIR}")
    print(f"Output folder: {REPORT_DIR}")

    # --------------------------------------------------------
    # Check required directories
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING DATASET DIRECTORIES")
    print("=" * 70)

    directories = [
        THICK_INFECTED,
        THICK_UNINFECTED,
        THIN_PARASITIZED,
        THIN_UNINFECTED,
        GT_ROOT,
    ]

    for directory in directories:

        status = "FOUND" if directory.exists() else "MISSING"

        print(
            f"{status:8} : {directory}"
        )

    # --------------------------------------------------------
    # Process thick classification dataset
    # --------------------------------------------------------

    process_directory(
        directory=THICK_INFECTED,
        smear_type="thick",
        label="infected",
        source="thick_classification",
    )

    process_directory(
        directory=THICK_UNINFECTED,
        smear_type="thick",
        label="uninfected",
        source="thick_classification",
    )

    # --------------------------------------------------------
    # Process thin classification dataset
    # --------------------------------------------------------

    process_directory(
        directory=THIN_PARASITIZED,
        smear_type="thin",
        label="parasitized",
        source="thin_classification",
    )

    process_directory(
        directory=THIN_UNINFECTED,
        smear_type="thin",
        label="uninfected",
        source="thin_classification",
    )

    # --------------------------------------------------------
    # Process patient-level thick-smear dataset
    # --------------------------------------------------------

    if GT_ROOT.exists():

        process_directory(
            directory=GT_ROOT,
            smear_type="thick",
            label="annotated",
            source="thick_patient_annotations",
        )

    # --------------------------------------------------------
    # Remove invalid images from master dataset
    # --------------------------------------------------------

    valid_records = [
        record
        for record in records
        if record["valid"]
    ]

    invalid_records = [
        record
        for record in records
        if not record["valid"]
    ]

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    hash_to_paths = {}

    for record in valid_records:

        file_hash = record["md5"]

        if not file_hash:
            continue

        hash_to_paths.setdefault(
            file_hash,
            [],
        ).append(record["image_path"])

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths in hash_to_paths.items()
        if len(paths) > 1
    }

    duplicate_image_count = sum(
        len(paths)
        for paths in duplicate_groups.values()
    )

    # --------------------------------------------------------
    # Save master dataset
    # --------------------------------------------------------

    output_file = (
        REPORT_DIR
        / "master_dataset.csv"
    )

    fieldnames = [
        "image_path",
        "filename",
        "smear_type",
        "label",
        "patient_id",
        "source",
        "width",
        "height",
        "mode",
        "format",
        "file_size_mb",
        "valid",
        "md5",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(valid_records)

    # --------------------------------------------------------
    # Save duplicate report
    # --------------------------------------------------------

    duplicate_file = (
        REPORT_DIR
        / "duplicate_images.csv"
    )

    with open(
        duplicate_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "md5",
                "image_path",
            ]
        )

        for file_hash, paths in duplicate_groups.items():

            for path in paths:

                writer.writerow(
                    [
                        file_hash,
                        path,
                    ]
                )

    # --------------------------------------------------------
    # Save invalid image report
    # --------------------------------------------------------

    invalid_file = (
        REPORT_DIR
        / "invalid_images.csv"
    )

    with open(
        invalid_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "filename",
                "smear_type",
                "label",
                "source",
                "width",
                "height",
                "mode",
                "format",
                "file_size_mb",
                "valid",
                "md5",
            ],
        )

        writer.writeheader()

        writer.writerows(
            invalid_records
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)

    print(
        f"Total records found       : {len(records):,}"
    )

    print(
        f"Valid images              : {len(valid_records):,}"
    )

    print(
        f"Invalid images            : {len(invalid_records):,}"
    )

    print(
        f"Duplicate groups          : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images involved in "
        f"duplicates               : "
        f"{duplicate_image_count:,}"
    )

    # --------------------------------------------------------
    # Smear statistics
    # --------------------------------------------------------

    print()
    print("Images by smear type:")

    smear_counts = Counter(
        record["smear_type"]
        for record in valid_records
    )

    for smear_type, count in smear_counts.items():

        print(
            f"  {smear_type:15} : {count:,}"
        )

    # --------------------------------------------------------
    # Label statistics
    # --------------------------------------------------------

    print()
    print("Images by label:")

    label_counts = Counter(
        record["label"]
        for record in valid_records
    )

    for label, count in label_counts.items():

        print(
            f"  {label:15} : {count:,}"
        )

    # --------------------------------------------------------
    # Source statistics
    # --------------------------------------------------------

    print()
    print("Images by source:")

    source_counts = Counter(
        record["source"]
        for record in valid_records
    )

    for source, count in source_counts.items():

        print(
            f"  {source:30} : {count:,}"
        )

    # --------------------------------------------------------
    # Patient statistics
    # --------------------------------------------------------

    patient_ids = {
        record["patient_id"]
        for record in valid_records
        if record["patient_id"] != "unknown"
    }

    print()
    print(
        f"Unique patient/sample IDs found: "
        f"{len(patient_ids):,}"
    )

    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Master dataset : {output_file}"
    )

    print(
        f"Duplicates     : {duplicate_file}"
    )

    print(
        f"Invalid images : {invalid_file}"
    )

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
