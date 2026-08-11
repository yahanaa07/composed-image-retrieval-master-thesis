import os
import json
import shutil
import concurrent.futures
from collections import defaultdict
from datetime import datetime

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

CSV_PATH = (
    "/home/users/ymathansingh/Master_Thesis/"
    "03_CIR-dataset/original/cir_db.csv"
)

IMAGES_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "01_download_images/png/downloaded_images_png"
)

OUTPUT_DIR = (
    "/home/users/ymathansingh/Master_Thesis/"
    "02_preprocessing/missing_images/"
    "results_missing_png"
)

UPDATED_CSV_DIR = (
    "/home/users/ymathansingh/Master_Thesis/"
    "03_CIR-dataset_updated/original"
)

OUTPUT_CSV = os.path.join(
    UPDATED_CSV_DIR,
    "cir_db_updated_png.csv"
)

PARTITION_JSON = os.path.join(
    OUTPUT_DIR,
    "partition_count.json"
)

VALIDATION_JSON = os.path.join(
    OUTPUT_DIR,
    "validation_summary.json"
)

# ------------------------------------------------------------
# Corrupt images are MOVED here.
# Original split structure is preserved.
# ------------------------------------------------------------

CORRUPT_IMAGES_ROOT = (
    "/scratch/users/ymathansingh/"
    "missing_images/corrupt_images_png"
)

SPLITS = [
    "train",
    "val",
    "test_id",
    "test_ood"
]

# Number of parallel workers for image validation.
# This is CPU/file-I/O only. No GPU is used.
NUM_THREADS = 12


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories():
    """Create all required output directories."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for split in SPLITS:
        os.makedirs(
            os.path.join(
                CORRUPT_IMAGES_ROOT,
                split
            ),
            exist_ok=True
        )


# ============================================================
# PARSE ID2
# ============================================================

def parse_id2(value):
    """
    Convert id2 into a list of image IDs.

    Supports:
        comma
        semicolon
        pipe
        whitespace

    Empty / NaN values return [].
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    # Normalize separators
    text = text.replace(";", ",")
    text = text.replace("|", ",")

    parts = [
        p.strip()
        for p in text.split(",")
        if p.strip()
    ]

    # If no comma-style separator was found,
    # fall back to whitespace.
    if len(parts) == 1 and " " in text:
        parts = [
            p.strip()
            for p in text.split()
            if p.strip()
        ]

    return parts


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index():
    """
    Scan the four dataset partitions once.

    Creates:

        image_index:
            image_id -> {
                "split": split,
                "path": full_path
            }

    Also records the number of png files physically
    present in every split.

    If the same image ID occurs in multiple splits,
    the first occurrence is retained.
    """

    image_index = {}
    folder_totals = {}
    duplicate_ids_on_disk = defaultdict(list)

    print()
    print("=" * 60)
    print("SCANNING IMAGE FOLDERS")
    print("=" * 60)

    for split in SPLITS:

        folder = os.path.join(
            IMAGES_ROOT,
            split
        )

        count = 0

        if not os.path.isdir(folder):

            print(
                f"WARNING: missing folder: {folder}"
            )

            folder_totals[split] = 0
            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(".png"):
                continue

            image_id = os.path.splitext(
                filename
            )[0]

            full_path = os.path.join(
                folder,
                filename
            )

            count += 1

            # Record all physical occurrences of an ID.
            duplicate_ids_on_disk[
                image_id
            ].append(
                {
                    "split": split,
                    "path": full_path
                }
            )

            # First occurrence wins.
            if image_id not in image_index:

                image_index[image_id] = {
                    "split": split,
                    "path": full_path
                }

        folder_totals[split] = count

        print(
            f"{split:<10}: "
            f"{count:>10,} png files"
        )

    print("-" * 60)

    print(
        f"{'total':<10}: "
        f"{sum(folder_totals.values()):>10,}"
    )

    print("=" * 60)

    duplicate_ids = {
        image_id: locations
        for image_id, locations
        in duplicate_ids_on_disk.items()
        if len(locations) > 1
    }

    return (
        image_index,
        folder_totals,
        duplicate_ids
    )


# ============================================================
# VALIDATE ONE IMAGE
# ============================================================

def validate_image(item):
    """
    Validate one image using PIL.

    Image.open() checks whether PIL can identify the file.

    img.load() forces the image data to actually be decoded,
    which catches files that have a valid-looking header but
    contain corrupted/unreadable image data.

    Returns:
        {
            image_id,
            split,
            path,
            valid,
            error
        }
    """

    image_id = item["image_id"]
    split = item["split"]
    path = item["path"]

    try:

        with Image.open(path) as img:

            # Force complete image decoding.
            img.load()

        return {
            "image_id": image_id,
            "split": split,
            "path": path,
            "valid": True,
            "error": None
        }

    except Exception as e:

        return {
            "image_id": image_id,
            "split": split,
            "path": path,
            "valid": False,
            "error": str(e)
        }


# ============================================================
# VALIDATE ALL PHYSICAL IMAGES
# ============================================================

def validate_all_images(image_index):
    """
    Validate every physically present png image.

    Returns:

        valid_images
        corrupt_images
        validation_errors
    """

    items = []

    for image_id, info in image_index.items():

        items.append(
            {
                "image_id": image_id,
                "split": info["split"],
                "path": info["path"]
            }
        )

    valid_images = {}
    corrupt_images = {}
    validation_errors = {}

    print()
    print("=" * 60)
    print("VALIDATING IMAGE CONTENT")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=NUM_THREADS
    ) as executor:

        results = executor.map(
            validate_image,
            items
        )

        for result in tqdm(
            results,
            total=len(items),
            desc="Validating png images"
        ):

            image_id = result["image_id"]

            if result["valid"]:

                valid_images[image_id] = result

            else:

                corrupt_images[image_id] = result

                validation_errors[
                    image_id
                ] = result["error"]

    print()
    print(
        f"Valid images   : "
        f"{len(valid_images):,}"
    )

    print(
        f"Corrupt images : "
        f"{len(corrupt_images):,}"
    )

    print("=" * 60)

    return (
        valid_images,
        corrupt_images,
        validation_errors
    )


# ============================================================
# MOVE CORRUPT IMAGES
# ============================================================

def move_corrupt_images(corrupt_images):
    """
    Move every physically corrupt png to:

        CORRUPT_IMAGES_ROOT/
            train/
            val/
            test_id/
            test_ood/

    Files are MOVED, not copied.
    """

    moved = {}
    move_errors = {}

    print()
    print("=" * 60)
    print("MOVING CORRUPT IMAGES")
    print("=" * 60)

    for image_id, info in tqdm(
        corrupt_images.items(),
        total=len(corrupt_images),
        desc="Moving corrupt images"
    ):

        split = info["split"]
        source_path = info["path"]

        destination_dir = os.path.join(
            CORRUPT_IMAGES_ROOT,
            split
        )

        os.makedirs(
            destination_dir,
            exist_ok=True
        )

        filename = os.path.basename(
            source_path
        )

        destination_path = os.path.join(
            destination_dir,
            filename
        )

        # Avoid overwriting an existing file.
        if os.path.exists(destination_path):

            base, extension = os.path.splitext(
                filename
            )

            counter = 1

            while os.path.exists(
                destination_path
            ):

                new_filename = (
                    f"{base}_corrupt_{counter}"
                    f"{extension}"
                )

                destination_path = os.path.join(
                    destination_dir,
                    new_filename
                )

                counter += 1

        try:

            shutil.move(
                source_path,
                destination_path
            )

            moved[image_id] = {
                "split": split,
                "source": source_path,
                "destination": destination_path,
                "error": info["error"]
            }

        except Exception as e:

            move_errors[image_id] = str(e)

    print()
    print(
        f"Corrupt images moved : "
        f"{len(moved):,}"
    )

    print(
        f"Move errors          : "
        f"{len(move_errors):,}"
    )

    print("=" * 60)

    return (
        moved,
        move_errors
    )


# ============================================================
# COLLECT CSV IMAGE IDs
# ============================================================

def collect_csv_ids(df):
    """
    Collect every image ID referenced by the CSV.

    Used only for reporting which corrupt images are actually
    relevant to the CIR dataset.
    """

    id1_ids = set()
    id2_ids = set()

    for value in df["id1"].dropna():

        image_id = str(value).strip()

        if image_id:
            id1_ids.add(image_id)

    for value in df["id2"].dropna():

        for image_id in parse_id2(value):

            id2_ids.add(image_id)

    return (
        id1_ids,
        id2_ids
    )


# ============================================================
# FILTER CIR ROWS
# ============================================================

def filter_cir_rows(
    df,
    image_index,
    valid_images
):
    """
    Keep a row only when:

        1. id1 exists physically
        2. id1 is a valid/readable image
        3. id2 is non-empty
        4. at least one id2 exists physically
        5. at least one id2 is a valid/readable image

    Returns:

        updated dataframe
        removal statistics
    """

    keep_rows = []

    removal_stats = {
        "id1_missing": 0,
        "id1_corrupt": 0,
        "id2_empty": 0,
        "id2_no_valid_target": 0
    }

    print()
    print("=" * 60)
    print("FILTERING CIR DATASET")
    print("=" * 60)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking CIR rows"
    ):

        # ----------------------------------------------------
        # ID1
        # ----------------------------------------------------

        id1 = str(
            row["id1"]
        ).strip()

        # ID1 does not exist at all.
        if id1 not in image_index:

            removal_stats[
                "id1_missing"
            ] += 1

            continue

        # ID1 exists physically but is corrupt.
        if id1 not in valid_images:

            removal_stats[
                "id1_corrupt"
            ] += 1

            continue

        # ----------------------------------------------------
        # ID2
        # ----------------------------------------------------

        id2_list = parse_id2(
            row["id2"]
        )

        # No target IDs.
        if len(id2_list) == 0:

            removal_stats[
                "id2_empty"
            ] += 1

            continue

        # ----------------------------------------------------
        # At least one valid target required.
        # ----------------------------------------------------

        valid_target_found = False

        for target_id in id2_list:

            # Target must exist.
            if target_id not in image_index:
                continue

            # Target must be readable.
            if target_id not in valid_images:
                continue

            valid_target_found = True
            break

        if not valid_target_found:

            removal_stats[
                "id2_no_valid_target"
            ] += 1

            continue

        # ----------------------------------------------------
        # All required criteria satisfied.
        # ----------------------------------------------------

        keep_rows.append(row)

    updated_df = pd.DataFrame(
        keep_rows,
        columns=df.columns
    )

    print()
    print(
        f"Original rows : "
        f"{len(df):,}"
    )

    print(
        f"Rows kept     : "
        f"{len(updated_df):,}"
    )

    print(
        f"Rows removed  : "
        f"{len(df) - len(updated_df):,}"
    )

    print()
    print("Removal reasons:")

    for reason, count in removal_stats.items():

        print(
            f"  {reason:<25}: "
            f"{count:>8,}"
        )

    print("=" * 60)

    return (
        updated_df,
        removal_stats
    )


# ============================================================
# SAVE PARTITION COUNTS
# ============================================================

def save_partition_counts(updated_df):
    """
    Count rows in each partition and save to JSON.
    """

    partition_counts = (
        updated_df[
            "partition"
        ]
        .value_counts()
        .to_dict()
    )

    with open(
        PARTITION_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            partition_counts,
            f,
            indent=2
        )

    print()
    print("Partition counts:")

    for partition in SPLITS:

        count = partition_counts.get(
            partition,
            0
        )

        print(
            f"  {partition:<10}: "
            f"{count:,}"
        )

    print()
    print(
        f"Partition count saved to:\n"
        f"{PARTITION_JSON}"
    )

    return partition_counts


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    # ========================================================
    # LOAD CSV
    # ========================================================

    print()
    print("=" * 60)
    print("LOADING CIR CSV")
    print("=" * 60)

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"CSV rows: "
        f"{len(df):,}"
    )

    if "partition" in df.columns:

        print(
            "Original partition counts:"
        )

        print(
            df[
                "partition"
            ]
            .value_counts()
            .to_dict()
        )

    # ========================================================
    # BUILD IMAGE INDEX
    # ========================================================

    (
        image_index,
        folder_totals_before,
        duplicate_ids_on_disk
    ) = build_image_index()

    print()
    print(
        f"Unique image IDs indexed: "
        f"{len(image_index):,}"
    )

    if duplicate_ids_on_disk:

        print(
            f"WARNING: "
            f"{len(duplicate_ids_on_disk):,} "
            f"image IDs occur in multiple split folders."
        )

    # ========================================================
    # VALIDATE PHYSICAL IMAGE CONTENT
    # ========================================================

    (
        valid_images,
        corrupt_images,
        validation_errors
    ) = validate_all_images(
        image_index
    )

    # ========================================================
    # COLLECT CSV IDS
    # ========================================================

    (
        csv_id1_ids,
        csv_id2_ids
    ) = collect_csv_ids(
        df
    )

    csv_combined_ids = (
        csv_id1_ids |
        csv_id2_ids
    )

    corrupt_referenced_ids = (
        set(corrupt_images.keys())
        &
        csv_combined_ids
    )

    # ========================================================
    # MOVE CORRUPT PHYSICAL FILES
    # ========================================================

    (
        moved_corrupt,
        corrupt_move_errors
    ) = move_corrupt_images(
        corrupt_images
    )

    # ========================================================
    # FILTER CIR CSV
    # ========================================================

    (
        updated_df,
        removal_stats
    ) = filter_cir_rows(
        df,
        image_index,
        valid_images
    )

    # ========================================================
    # SAVE UPDATED CSV
    # ========================================================

    updated_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print(
        f"Updated CSV saved to:\n"
        f"{OUTPUT_CSV}"
    )

    # ========================================================
    # PARTITION COUNTS
    # ========================================================

    partition_counts = save_partition_counts(
        updated_df
    )

    # ========================================================
    # FINAL PHYSICAL IMAGE COUNTS
    # ========================================================

    (
        _,
        folder_totals_after,
        _
    ) = build_image_index()

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    summary = {

        "generated_at": (
            datetime.now().isoformat()
        ),

        "csv": {
            "input": CSV_PATH,
            "output": OUTPUT_CSV,
            "original_rows": len(df),
            "updated_rows": len(updated_df),
            "rows_removed": (
                len(df) -
                len(updated_df)
            )
        },

        "images": {
            "root": IMAGES_ROOT,
            "splits": SPLITS,
            "physical_files_before": (
                folder_totals_before
            ),
            "physical_files_after": (
                folder_totals_after
            ),
            "unique_ids_indexed": (
                len(image_index)
            ),
            "valid_images": (
                len(valid_images)
            ),
            "corrupt_images": (
                len(corrupt_images)
            )
        },

        "corrupt_images": {
            "total": len(corrupt_images),
            "referenced_by_cir_csv": (
                len(corrupt_referenced_ids)
            ),
            "moved": len(moved_corrupt),
            "move_errors": len(
                corrupt_move_errors
            ),
            "destination": (
                CORRUPT_IMAGES_ROOT
            )
        },

        "csv_id_inventory": {
            "unique_id1": len(
                csv_id1_ids
            ),
            "unique_id2": len(
                csv_id2_ids
            ),
            "unique_combined": len(
                csv_combined_ids
            )
        },

        "row_removal_reasons": (
            removal_stats
        ),

        "partition_counts": (
            partition_counts
        )
    }

    with open(
        VALIDATION_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Original CSV rows      : "
        f"{len(df):,}"
    )

    print(
        f"Final CSV rows         : "
        f"{len(updated_df):,}"
    )

    print(
        f"Rows removed           : "
        f"{len(df) - len(updated_df):,}"
    )

    print()

    print(
        f"Valid images           : "
        f"{len(valid_images):,}"
    )

    print(
        f"Corrupt images found   : "
        f"{len(corrupt_images):,}"
    )

    print(
        f"Corrupt images moved   : "
        f"{len(moved_corrupt):,}"
    )

    print(
        f"Corrupt move errors    : "
        f"{len(corrupt_move_errors):,}"
    )

    print()
    print("Files created:")
    print(
        f"  {OUTPUT_CSV}"
    )
    print(
        f"  {PARTITION_JSON}"
    )
    print(
        f"  {VALIDATION_JSON}"
    )

    print()
    print("Corrupt images moved to:")
    print(
        f"  {CORRUPT_IMAGES_ROOT}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
