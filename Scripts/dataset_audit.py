"""
Dataset Audit
-------------
Audits the malaria blood-smear dataset before preprocessing/training.

Expected project structure:

project/
├── Scripts/
│   └── dataset_audit.py
├── data/
│   ├── thick_smears/
│   │   ├── 150_patient_thick_annoted_blood_smears/
│   │   │   └── GT_updated/
│   │   │       ├── TF100_CS39/
│   │   │       └── ...
│   │   ├── Thick_Infected/
│   │   └── Thick_Uninfected/
│   │
│   └── thin_smear/
│       ├── cell_images/
│       │   ├── Parasitized/
│       │   └── Uninfected/
│       ├── Parasitized/
│       └── Uninfected/
└── reports/
"""

from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import hashlib
import csv


# ============================================================
# CONFIGURATION
# ============================================================

# dataset_audit.py is inside Scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def get_image_files(directory):
    """Return all image files recursively under a directory."""
    if not directory.exists():
        return []

    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_file_size_mb(path):
    """Return file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def calculate_md5(path, chunk_size=1024 * 1024):
    """Calculate MD5 hash for duplicate detection."""
    md5 = hashlib.md5()

    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return None


def inspect_image(path):
    """
    Try opening an image and collect basic metadata.

    Returns:
        dict containing image information.
    """

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            image_format = img.format

            return {
                "valid": True,
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "error": "",
            }

    except Exception as e:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "error": str(e),
        }


def print_header(title):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATASET LOCATIONS
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
# AUDIT STORAGE
# ============================================================

all_records = []
corrupt_images = []
duplicate_hashes = defaultdict(list)

extension_counter = Counter()
format_counter = Counter()
dimension_counter = Counter()
mode_counter = Counter()

total_valid = 0
total_invalid = 0


# ============================================================
# AUDIT FUNCTION
# ============================================================

def audit_directory(directory, dataset_name, label=None):
    """
    Audit all images inside a directory.
    """

    global total_valid
    global total_invalid

    images = get_image_files(directory)

    print_header(f"{dataset_name}")

    print(f"Directory : {directory}")
    print(f"Exists    : {directory.exists()}")
    print(f"Images    : {len(images):,}")

    if not images:
        print("No images found.")
        return

    for image_path in images:

        info = inspect_image(image_path)

        relative_path = image_path.relative_to(PROJECT_ROOT)

        extension = image_path.suffix.lower()
        extension_counter[extension] += 1

        record = {
            "dataset": dataset_name,
            "label": label or "",
            "path": str(relative_path),
            "filename": image_path.name,
            "extension": extension,
            "size_mb": round(get_file_size_mb(image_path), 4),
            "valid": info["valid"],
            "width": info["width"],
            "height": info["height"],
            "mode": info["mode"],
            "format": info["format"],
            "error": info["error"],
        }

        all_records.append(record)

        if info["valid"]:

            total_valid += 1

            format_counter[info["format"]] += 1
            mode_counter[info["mode"]] += 1

            dimension_counter[
                (info["width"], info["height"])
            ] += 1

            # Duplicate detection
            file_hash = calculate_md5(image_path)

            if file_hash:
                duplicate_hashes[file_hash].append(str(relative_path))

        else:

            total_invalid += 1
            corrupt_images.append(record)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MALARIA BLOOD-SMEAR DATASET AUDIT")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data folder  : {DATA_DIR}")
    print(f"Report folder: {REPORT_DIR}")

    # --------------------------------------------------------
    # Check important directories
    # --------------------------------------------------------

    print_header("DIRECTORY CHECK")

    directories = {
        "Data": DATA_DIR,
        "Thick smears": THICK_ROOT,
        "Thin smear": THIN_ROOT,
        "Thick infected": THICK_INFECTED,
        "Thick uninfected": THICK_UNINFECTED,
        "Thick patient dataset": THICK_PATIENT_ROOT,
        "Ground truth": GT_ROOT,
        "Thin parasitized": THIN_PARASITIZED,
        "Thin uninfected": THIN_UNINFECTED,
    }

    for name, path in directories.items():

        status = "FOUND" if path.exists() else "MISSING"

        print(f"{name:25} : {status}")

    # --------------------------------------------------------
    # Thick smear classification folders
    # --------------------------------------------------------

    audit_directory(
        THICK_INFECTED,
        "THICK - INFECTED",
        "infected",
    )

    audit_directory(
        THICK_UNINFECTED,
        "THICK - UNINFECTED",
        "uninfected",
    )

    # --------------------------------------------------------
    # Thin smear classification folders
    # --------------------------------------------------------

    audit_directory(
        THIN_PARASITIZED,
        "THIN - PARASITIZED",
        "parasitized",
    )

    audit_directory(
        THIN_UNINFECTED,
        "THIN - UNINFECTED",
        "uninfected",
    )

    # --------------------------------------------------------
    # Thick patient-level dataset
    # --------------------------------------------------------

    if THICK_PATIENT_ROOT.exists():

        print_header("THICK PATIENT-LEVEL DATASET")

        patient_dirs = [
            p
            for p in GT_ROOT.iterdir()
            if p.is_dir()
        ] if GT_ROOT.exists() else []

        print(f"Patient/sample folders found: {len(patient_dirs):,}")

        patient_summary = []

        for patient_dir in sorted(patient_dirs):

            images = get_image_files(patient_dir)

            patient_summary.append(
                {
                    "patient": patient_dir.name,
                    "images": len(images),
                }
            )

        for item in patient_summary:

            print(
                f"{item['patient']:25} "
                f": {item['images']:5,} images"
            )

        # Save patient summary
        patient_csv = REPORT_DIR / "patient_summary.csv"

        with open(
            patient_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=["patient", "images"],
            )

            writer.writeheader()
            writer.writerows(patient_summary)

        print()
        print(f"Patient summary saved to: {patient_csv}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header("DATASET SUMMARY")

    print(f"Total image files found : {len(all_records):,}")
    print(f"Valid images            : {total_valid:,}")
    print(f"Invalid/corrupt images  : {total_invalid:,}")

    print()
    print("Images by dataset:")

    dataset_counts = Counter(
        record["dataset"]
        for record in all_records
    )

    for dataset, count in dataset_counts.items():

        print(f"  {dataset:30} : {count:,}")

    print()
    print("Images by label:")

    label_counts = Counter(
        record["label"]
        for record in all_records
        if record["label"]
    )

    for label, count in label_counts.items():

        print(f"  {label:30} : {count:,}")

    # --------------------------------------------------------
    # File formats
    # --------------------------------------------------------

    print_header("FILE EXTENSIONS")

    for extension, count in extension_counter.most_common():

        print(f"{extension:10} : {count:,}")

    print_header("IMAGE FORMATS")

    for image_format, count in format_counter.most_common():

        print(f"{str(image_format):10} : {count:,}")

    # --------------------------------------------------------
    # Image modes
    # --------------------------------------------------------

    print_header("IMAGE COLOR MODES")

    for mode, count in mode_counter.most_common():

        print(f"{mode:10} : {count:,}")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print_header("IMAGE DIMENSIONS")

    for dimension, count in dimension_counter.most_common(20):

        width, height = dimension

        print(
            f"{width:5} x {height:<5} : {count:,}"
        )

    if len(dimension_counter) > 20:

        print(
            f"... and "
            f"{len(dimension_counter) - 20:,} more dimensions"
        )

    # --------------------------------------------------------
    # Corrupt images
    # --------------------------------------------------------

    print_header("CORRUPT / INVALID IMAGES")

    if not corrupt_images:

        print("No corrupt images detected.")

    else:

        print(
            f"Found {len(corrupt_images):,} "
            "corrupt/invalid images:"
        )

        for record in corrupt_images[:50]:

            print(f"  {record['path']}")
            print(f"    Error: {record['error']}")

        if len(corrupt_images) > 50:

            print(
                f"... and "
                f"{len(corrupt_images) - 50:,} more"
            )

    # --------------------------------------------------------
    # Duplicate images
    # --------------------------------------------------------

    print_header("DUPLICATE IMAGES")

    duplicate_groups = [
        paths
        for paths in duplicate_hashes.values()
        if len(paths) > 1
    ]

    print(
        f"Duplicate groups found: "
        f"{len(duplicate_groups):,}"
    )

    if duplicate_groups:

        for index, group in enumerate(
            duplicate_groups[:20],
            start=1,
        ):

            print()
            print(f"Group {index}:")

            for path in group:

                print(f"  {path}")

        if len(duplicate_groups) > 20:

            print(
                f"\n... and "
                f"{len(duplicate_groups) - 20:,} "
                "more duplicate groups"
            )

    # --------------------------------------------------------
    # Save complete CSV
    # --------------------------------------------------------

    csv_path = REPORT_DIR / "dataset_audit.csv"

    fieldnames = [
        "dataset",
        "label",
        "path",
        "filename",
        "extension",
        "size_mb",
        "valid",
        "width",
        "height",
        "mode",
        "format",
        "error",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_records)

    print_header("AUDIT COMPLETE")

    print(f"Detailed CSV report:")
    print(csv_path)

    print()
    print("Next step:")
    print(
        "Review the output before we build "
        "the dataset-preparation pipeline."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
