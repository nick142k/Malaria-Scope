"""
audit_thick.py
==============

Audit the baseline thick-smear binary classification dataset.

Dataset:
    data/thick_smears/Thick_Infected/
    data/thick_smears/Thick_Uninfected/

This script:
    - counts images
    - checks image readability
    - reports dimensions
    - reports formats
    - reports duplicate MD5 hashes
    - checks duplicate filenames across classes

IMPORTANT:
    Original images are never modified.
"""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib

from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THICK_ROOT = PROJECT_ROOT / "data" / "thick_smears"

INFECTED_DIR = THICK_ROOT / "Thick_Infected"
UNINFECTED_DIR = THICK_ROOT / "Thick_Uninfected"

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
# HELPERS
# ============================================================

def get_images(directory):
    """Return supported image files recursively."""

    return sorted(
        [
            p
            for p in directory.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


def md5(path):
    """Calculate MD5 hash for duplicate detection."""

    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def audit_images(images, class_name):
    """Audit a collection of images."""

    dimensions = Counter()
    formats = Counter()
    modes = Counter()

    hashes = defaultdict(list)

    corrupt = []

    for index, path in enumerate(images, start=1):

        try:

            with Image.open(path) as image:

                image.verify()

            with Image.open(path) as image:

                dimensions[
                    image.size
                ] += 1

                formats[
                    image.format
                ] += 1

                modes[
                    image.mode
                ] += 1

            file_hash = md5(path)

            hashes[file_hash].append(
                str(path)
            )

        except Exception as error:

            corrupt.append(
                {
                    "path": str(path),
                    "error": str(error),
                }
            )

        if index % 250 == 0:

            print(
                f"{class_name}: "
                f"{index:,}/{len(images):,}"
            )

    duplicate_groups = {
        h: paths
        for h, paths in hashes.items()
        if len(paths) > 1
    }

    return {
        "dimensions": dimensions,
        "formats": formats,
        "modes": modes,
        "hashes": hashes,
        "duplicates": duplicate_groups,
        "corrupt": corrupt,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THICK SMEAR DATASET AUDIT")
    print("=" * 70)

    infected = get_images(
        INFECTED_DIR
    )

    uninfected = get_images(
        UNINFECTED_DIR
    )

    print()
    print(
        f"Infected images   : "
        f"{len(infected):,}"
    )

    print(
        f"Uninfected images : "
        f"{len(uninfected):,}"
    )

    print(
        f"Total images      : "
        f"{len(infected) + len(uninfected):,}"
    )

    # --------------------------------------------------------
    # Audit infected
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDITING INFECTED")
    print("=" * 70)

    infected_audit = audit_images(
        infected,
        "Infected",
    )

    # --------------------------------------------------------
    # Audit uninfected
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDITING UNINFECTED")
    print("=" * 70)

    uninfected_audit = audit_images(
        uninfected,
        "Uninfected",
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIMENSIONS")
    print("=" * 70)

    print()
    print("INFECTED:")

    for dimension, count in (
        infected_audit["dimensions"]
        .most_common()
    ):
        print(
            f"  {dimension[0]} x {dimension[1]}"
            f" : {count:,}"
        )

    print()
    print("UNINFECTED:")

    for dimension, count in (
        uninfected_audit["dimensions"]
        .most_common()
    ):
        print(
            f"  {dimension[0]} x {dimension[1]}"
            f" : {count:,}"
        )

    # --------------------------------------------------------
    # Formats
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FORMATS")
    print("=" * 70)

    print(
        "Infected:",
        dict(infected_audit["formats"])
    )

    print(
        "Uninfected:",
        dict(uninfected_audit["formats"])
    )

    # --------------------------------------------------------
    # Modes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE MODES")
    print("=" * 70)

    print(
        "Infected:",
        dict(infected_audit["modes"])
    )

    print(
        "Uninfected:",
        dict(uninfected_audit["modes"])
    )

    # --------------------------------------------------------
    # Corrupt files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CORRUPT / UNREADABLE IMAGES")
    print("=" * 70)

    corrupt = (
        infected_audit["corrupt"]
        + uninfected_audit["corrupt"]
    )

    print(
        f"Total corrupt: {len(corrupt):,}"
    )

    for item in corrupt[:20]:

        print(
            f"  {item['path']}"
        )

        print(
            f"    ERROR: {item['error']}"
        )

    # --------------------------------------------------------
    # Duplicate hashes within classes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXACT DUPLICATES")
    print("=" * 70)

    infected_duplicates = (
        infected_audit["duplicates"]
    )

    uninfected_duplicates = (
        uninfected_audit["duplicates"]
    )

    print(
        f"Infected duplicate groups   : "
        f"{len(infected_duplicates):,}"
    )

    print(
        f"Uninfected duplicate groups : "
        f"{len(uninfected_duplicates):,}"
    )

    # --------------------------------------------------------
    # Cross-class duplicate hashes
    # --------------------------------------------------------

    infected_hashes = (
        set(
            infected_audit["hashes"].keys()
        )
    )

    uninfected_hashes = (
        set(
            uninfected_audit["hashes"].keys()
        )
    )

    cross_class_hashes = (
        infected_hashes
        & uninfected_hashes
    )

    print()
    print(
        f"Cross-class duplicate hashes : "
        f"{len(cross_class_hashes):,}"
    )

    if cross_class_hashes:

        print()
        print(
            "WARNING: identical image content "
            "appears in both classes."
        )

        for file_hash in list(
            cross_class_hashes
        )[:20]:

            print()
            print(
                f"MD5: {file_hash}"
            )

            for path in (
                infected_audit["hashes"][file_hash]
            ):

                print(
                    f"  INFECTED   : {path}"
                )

            for path in (
                uninfected_audit["hashes"][file_hash]
            ):

                print(
                    f"  UNINFECTED : {path}"
                )

    # --------------------------------------------------------
    # Filename overlap
    # --------------------------------------------------------

    infected_names = Counter(
        p.name.lower()
        for p in infected
    )

    uninfected_names = Counter(
        p.name.lower()
        for p in uninfected
    )

    common_names = (
        set(infected_names)
        & set(uninfected_names)
    )

    print()
    print("=" * 70)
    print("FILENAME OVERLAP")
    print("=" * 70)

    print(
        f"Common filenames: "
        f"{len(common_names):,}"
    )

    for name in list(
        common_names
    )[:20]:

        print(
            f"  {name}"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
