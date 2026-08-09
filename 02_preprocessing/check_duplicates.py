import os
import json
import hashlib
import shutil
from collections import defaultdict
from datetime import datetime
import concurrent.futures

from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

IMAGES_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "01_download_images/jpg/downloaded_images_jpg"
)

"""
RESULT_ROOT = (
    "/scratch/users/ymathansingh/"
    "duplicate_images/results_duplicates_jpg"
)
"""
# JSON report location in HOME
RESULT_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "02_preprocessing/duplicate_images/"
    "results_duplicates_jpg"
)

# Duplicate images are moved here

REMOVED_DUPLICATES_ROOT = (
    "/scratch/users/ymathansingh/"
    "duplicate_images/results_duplicates_jpg/removed_duplicates"
)

SPLITS = ["train", "val", "test_id", "test_ood"]


# ============================================================
# PARAMETERS
# ============================================================

NUM_THREADS = 12


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_dirs():
    os.makedirs(
        RESULT_ROOT,
        exist_ok=True
    )

    for split in SPLITS:
        os.makedirs(
            os.path.join(
                REMOVED_DUPLICATES_ROOT,
                split
            ),
            exist_ok=True
        )


# ============================================================
# IMAGE LISTING
# ============================================================

def list_images(split):
    """Return all jpg images belonging to one dataset split."""

    folder = os.path.join(
        IMAGES_ROOT,
        split
    )

    images = []

    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        return images

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            img_id = os.path.splitext(
                filename
            )[0]

            images.append({
                "split": split,
                "id": img_id,
                "path": os.path.join(
                    folder,
                    filename
                )
            })

    return images


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images():
    """Count jpg images currently present in each dataset split."""

    counts = {}

    for split in SPLITS:

        folder = os.path.join(
            IMAGES_ROOT,
            split
        )

        if not os.path.isdir(folder):

            counts[split] = 0
            continue

        counts[split] = sum(
            1
            for filename in os.listdir(folder)
            if filename.lower().endswith(".jpg")
        )

    counts["total"] = sum(
        counts[split]
        for split in SPLITS
    )

    return counts


# ============================================================
# PRINT DATASET COUNTS
# ============================================================

def print_dataset_counts(title, counts):

    print()
    print("================================")
    print(title)
    print("================================")

    for split in SPLITS:

        print(
            f"{split:10s}: "
            f"{counts[split]:,}"
        )

    print("--------------------------------")

    print(
        f"{'total':10s}: "
        f"{counts['total']:,}"
    )

    print("================================")


# ============================================================
# MD5 HASHING
# ============================================================

def compute_md5(image):
    """
    Compute MD5 from RGB pixel values.

    This detects exact image-content duplicates,
    independent of filename.
    """

    try:

        with Image.open(
            image["path"]
        ) as img:

            img = img.convert("RGB")

            digest = hashlib.md5(
                img.tobytes()
            ).hexdigest()

        return {
            "split": image["split"],
            "id": image["id"],
            "hash": digest,
            "error": None
        }

    except Exception as e:

        return {
            "split": image["split"],
            "id": image["id"],
            "hash": None,
            "error": str(e)
        }


# ============================================================
# PARALLEL HASH INDEX
# ============================================================

def build_hash_index(
    images,
    function,
    description
):
    """
    Compute hashes in parallel.

    Uses executor.map() instead of creating hundreds
    of thousands of individual futures.
    """

    index = defaultdict(list)
    errors = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=NUM_THREADS
    ) as executor:

        results = executor.map(
            function,
            images
        )

        for result in tqdm(
            results,
            total=len(images),
            desc=description
        ):

            split = result["split"]
            img_id = result["id"]
            digest = result["hash"]
            error = result["error"]

            if error is not None:

                errors[
                    f"{split}:{img_id}"
                ] = error

                continue

            index[digest].append({
                "split": split,
                "id": img_id
            })

    return index, errors


# ============================================================
# ID COLLISION CHECK
# ============================================================

def check_id_collisions(images):
    """Find IDs occurring more than once."""

    id_groups = defaultdict(list)

    for img in images:

        id_groups[
            img["id"].lower()
        ].append(img)

    return {
        img_id: items
        for img_id, items in id_groups.items()
        if len(items) > 1
    }


# ============================================================
# VALIDATE ID COLLISIONS
# ============================================================

def validate_id_collisions(
    collisions,
    md5_index
):
    """
    Separate repeated IDs into:

    1. Same ID + same image content
    2. Same ID + different image content

    Images whose MD5 failed are placed into
    unable_to_validate.
    """

    md5_lookup = {}

    for md5, items in md5_index.items():

        for item in items:

            key = (
                item["split"],
                item["id"]
            )

            md5_lookup[key] = md5

    same_content = {}
    different_content = {}
    unable_to_validate = {}

    for img_id, images in collisions.items():

        hashes = set()
        missing_hash = False

        for img in images:

            key = (
                img["split"],
                img["id"]
            )

            if key not in md5_lookup:

                missing_hash = True
                continue

            hashes.add(
                md5_lookup[key]
            )

        if missing_hash:

            unable_to_validate[
                img_id
            ] = images

        elif len(hashes) == 1:

            same_content[
                img_id
            ] = images

        else:

            different_content[
                img_id
            ] = images

    return (
        same_content,
        different_content,
        unable_to_validate
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

def find_exact_duplicates(md5_index):
    """
    Find all MD5 groups containing more than one image.

    These groups are the single source of truth for
    duplicate removal.
    """

    return {
        md5: items
        for md5, items in md5_index.items()
        if len(items) > 1
    }


# ============================================================
# DUPLICATE KEY
# ============================================================

def image_key(image):
    """Unique identifier based on partition and filename/ID."""

    return (
        image["split"],
        image["id"]
    )


# ============================================================
# COLLECT DUPLICATES TO MOVE
# ============================================================

def collect_duplicates_to_move(
    exact_duplicates
):
    """
    Use exact MD5 duplicate groups as the single source
    of truth for deciding which images to remove.

    For every exact duplicate group:
        - Keep the first image.
        - Move all remaining images.

    This handles:
        - Same ID + same content
        - Different ID + same content
        - Duplicates within a partition
        - Duplicates across partitions

    Same ID + different content is NOT included because
    those images have different MD5 hashes and therefore
    are not part of the same exact-content duplicate group.
    """

    duplicates_to_move = {}
    kept_images = {}

    # --------------------------------------------------------
    # Exact MD5 duplicate groups
    # --------------------------------------------------------

    for md5, images in exact_duplicates.items():

        if len(images) <= 1:
            continue

        # ----------------------------------------------------
        # Keep the first image as representative
        # ----------------------------------------------------

        keeper = images[0]

        kept_key = image_key(
            keeper
        )

        kept_images[kept_key] = {
            "split": keeper["split"],
            "id": keeper["id"],
            "reason": "exact_content_duplicate",
            "md5": md5
        }

        # ----------------------------------------------------
        # Move all remaining copies
        # ----------------------------------------------------

        for duplicate in images[1:]:

            key = image_key(
                duplicate
            )

            if key not in duplicates_to_move:

                duplicates_to_move[key] = {
                    "split": duplicate["split"],
                    "id": duplicate["id"],
                    "source_path": None,
                    "methods": []
                }

            duplicates_to_move[
                key
            ]["methods"].append(
                "exact_content_duplicate"
            )

    return (
        duplicates_to_move,
        kept_images
    )


# ============================================================
# MOVE DUPLICATES
# ============================================================

def move_duplicates(
    duplicates_to_move,
    all_images
):
    """
    Move detected duplicate images from IMAGES_ROOT
    into REMOVED_DUPLICATES_ROOT.

    The original partition structure is preserved:

        removed_duplicates/
            train/
            val/
            test_id/
            test_ood/

    Files are moved, not copied.
    """

    image_lookup = {}

    for image in all_images:

        image_lookup[
            image_key(image)
        ] = image["path"]

    moved = {}
    move_errors = {}

    print()
    print("================================")
    print("Moving duplicate images")
    print("================================")

    for key, duplicate in tqdm(
        duplicates_to_move.items(),
        total=len(duplicates_to_move),
        desc="Moving duplicates"
    ):

        split = duplicate["split"]
        img_id = duplicate["id"]

        source_path = image_lookup.get(
            key
        )

        if source_path is None:

            move_errors[
                f"{split}:{img_id}"
            ] = "Source image not found"

            continue

        if not os.path.isfile(
            source_path
        ):

            move_errors[
                f"{split}:{img_id}"
            ] = (
                f"Source file does not exist: "
                f"{source_path}"
            )

            continue

        destination_dir = os.path.join(
            REMOVED_DUPLICATES_ROOT,
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

        if os.path.exists(
            destination_path
        ):

            base, extension = os.path.splitext(
                filename
            )

            counter = 1

            while os.path.exists(
                destination_path
            ):

                new_filename = (
                    f"{base}_duplicate_{counter}"
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

            moved[key] = {
                "split": split,
                "id": img_id,
                "source": source_path,
                "destination": destination_path,
                "methods": duplicate["methods"]
            }

        except Exception as e:

            move_errors[
                f"{split}:{img_id}"
            ] = str(e)

    return (
        moved,
        move_errors
    )


# ============================================================
# JSON-SAFE MOVED FILES
# ============================================================

def make_moved_files_json_safe(
    moved_duplicates
):
    """
    Convert tuple keys used internally by moved_duplicates
    into string keys so the dictionary can be written as JSON.

    Internal duplicate detection and moving logic is unchanged.
    """

    return {
        f"{key[0]}:{key[1]}": value
        for key, value in moved_duplicates.items()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    # --------------------------------------------------------
    # INITIAL DATASET COUNTS
    # --------------------------------------------------------

    initial_counts = count_images()

    print_dataset_counts(
        "Initial dataset image counts",
        initial_counts
    )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    all_images = []

    for split in SPLITS:

        images = list_images(
            split
        )

        all_images.extend(
            images
        )

    # --------------------------------------------------------
    # MD5
    # --------------------------------------------------------

    print()
    print("Calculating MD5...")

    md5_index, md5_errors = build_hash_index(
        all_images,
        compute_md5,
        "MD5 hashing"
    )

    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    exact_duplicates = find_exact_duplicates(
        md5_index
    )

    # --------------------------------------------------------
    # ID CHECK
    # --------------------------------------------------------

    print()
    print("Checking IDs...")

    id_collisions = check_id_collisions(
        all_images
    )

    (
        id_same_content,
        id_diff_content,
        id_unable_to_validate
    ) = validate_id_collisions(
        id_collisions,
        md5_index
    )

    # --------------------------------------------------------
    # COLLECT DUPLICATES TO MOVE
    # --------------------------------------------------------

    print()
    print("Identifying files to move...")

    (
        duplicates_to_move,
        kept_images
    ) = collect_duplicates_to_move(
        exact_duplicates
    )

    print(
        f"Duplicate files to move: "
        f"{len(duplicates_to_move):,}"
    )

    print(
        f"Files kept as representatives: "
        f"{len(kept_images):,}"
    )

    # --------------------------------------------------------
    # MOVE DUPLICATES
    # --------------------------------------------------------

    moved_duplicates, move_errors = move_duplicates(
        duplicates_to_move,
        all_images
    )

    # --------------------------------------------------------
    # FINAL DATASET COUNTS AFTER MOVING DUPLICATES
    # --------------------------------------------------------

    final_counts = count_images()

    print_dataset_counts(
        "Final dataset image counts after duplicate removal",
        final_counts
    )

    # --------------------------------------------------------
    # MAKE MOVED FILES JSON SAFE
    # --------------------------------------------------------

    moved_duplicates_json = make_moved_files_json_safe(
        moved_duplicates
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "initial_image_counts": initial_counts,

        "final_image_counts": final_counts,

        "generated_at": datetime.now().isoformat(),

        "parameters": {
            "num_threads": NUM_THREADS
        },

        "id_check": {
            "same_id_same_content": (
                id_same_content
            ),

            "same_id_different_content": (
                id_diff_content
            ),

            "same_id_unable_to_validate": (
                id_unable_to_validate
            )
        },

        "exact_content_duplicates": (
            exact_duplicates
        ),

        "duplicate_removal": {

            "description": (
                "Exact RGB-content duplicates are "
                "identified globally using MD5. "
                "One representative image is kept "
                "from each exact duplicate group, "
                "and all remaining copies are moved "
                "to removed_duplicates. "
                "Same-ID/different-content cases are "
                "reported as ID collisions and are "
                "not removed."
            ),

            "files_identified_for_moving": (
                len(duplicates_to_move)
            ),

            "files_successfully_moved": (
                len(moved_duplicates)
            ),

            "files_kept_as_representatives": (
                len(kept_images)
            ),

            "moved_files": (
                moved_duplicates_json
            ),

            "move_errors": (
                move_errors
            )
        },

        "errors": {
            "md5": md5_errors
        }
    }

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    output_file = os.path.join(
        RESULT_ROOT,
        "duplicate_analysis_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("================================")
    print("Duplicate analysis completed")
    print("================================")

    print(
        f"Images processed        : "
        f"{len(all_images):,}"
    )

    print(
        f"MD5 errors              : "
        f"{len(md5_errors):,}"
    )

    print(
        f"ID collisions           : "
        f"{len(id_collisions):,}"
    )

    print(
        f"Exact duplicate groups  : "
        f"{len(exact_duplicates):,}"
    )

    print(
        f"Files to move           : "
        f"{len(duplicates_to_move):,}"
    )

    print(
        f"Files successfully moved: "
        f"{len(moved_duplicates):,}"
    )

    print(
        f"Move errors             : "
        f"{len(move_errors):,}"
    )

    print()
    print("JSON report saved to:")
    print(output_file)

    print()
    print("Duplicate images moved to:")
    print(REMOVED_DUPLICATES_ROOT)

    print("================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()