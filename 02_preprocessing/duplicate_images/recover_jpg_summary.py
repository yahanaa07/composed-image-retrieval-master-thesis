import os
import json
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

ORIGINAL_IMAGES = (
    "/home/users/ymathansingh/Master_Thesis/"
    "01_download_images/jpg/downloaded_images_jpg"
)

REMOVED_DUPLICATES = (
    "/scratch/users/ymathansingh/"
    "duplicate_images/results_duplicates_jpg/removed_duplicates"
)

JSON_REPORT_DIR = (
    "/home/users/ymathansingh/Master_Thesis/"
    "02_preprocessing/duplicate_images/results_duplicates_jpg"
)

OUTPUT_JSON = os.path.join(
    JSON_REPORT_DIR,
    "jpg_duplicate_analysis_report.json"
)

SPLITS = ["train", "val", "test_id", "test_ood"]

# ============================================================
# COUNT HELPER
# ============================================================

def count_jpg(folder):
    """Count .jpg files in a folder."""

    if not os.path.isdir(folder):
        return 0

    return sum(
        1
        for filename in os.listdir(folder)
        if filename.lower().endswith(".jpg")
    )

# ============================================================
# CURRENT COUNTS IN ORIGINAL LOCATION
# ============================================================

final_counts = {}

for split in SPLITS:
    final_counts[split] = count_jpg(
        os.path.join(ORIGINAL_IMAGES, split)
    )

final_counts["total"] = sum(final_counts.values())

# ============================================================
# COUNTS OF REMOVED DUPLICATES
# ============================================================

removed_counts = {}

for split in SPLITS:
    removed_counts[split] = count_jpg(
        os.path.join(REMOVED_DUPLICATES, split)
    )

removed_counts["total"] = sum(removed_counts.values())

# ============================================================
# RECONSTRUCT INITIAL COUNTS
# ============================================================

initial_counts = {}

for split in SPLITS:
    initial_counts[split] = (
        final_counts[split] + removed_counts[split]
    )

initial_counts["total"] = sum(initial_counts.values())

# ============================================================
# BUILD REPORT
# ============================================================

report = {
    "report_type": "recovered_after_duplicate_removal",
    "generated_at": datetime.now().isoformat(),

    "paths": {
        "original_images": ORIGINAL_IMAGES,
        "removed_duplicates": REMOVED_DUPLICATES,
        "json_report": JSON_REPORT_DIR,
    },

    "initial_image_counts_reconstructed": initial_counts,

    "final_image_counts_in_original_location": final_counts,

    "removed_duplicate_counts": removed_counts,

    "duplicate_removal_summary": {
        "images_removed_total": removed_counts["total"],
        "images_remaining_total": final_counts["total"],
        "images_before_removal_total": initial_counts["total"],

        "verification": {
            "before_equals_remaining_plus_removed": (
                initial_counts["total"]
                == final_counts["total"] + removed_counts["total"]
            )
        }
    }
}

# ============================================================
# SAVE REPORT
# ============================================================

os.makedirs(JSON_REPORT_DIR, exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("========================================")
print("Recovered JPG duplicate summary created")
print("========================================")
print(f"Output JSON: {OUTPUT_JSON}")
print()

print("Initial reconstructed counts:")
for split in SPLITS:
    print(f"  {split:10s}: {initial_counts[split]:,}")
print(f"  total     : {initial_counts['total']:,}")
print()

print("Final remaining counts:")
for split in SPLITS:
    print(f"  {split:10s}: {final_counts[split]:,}")
print(f"  total     : {final_counts['total']:,}")
print()

print("Removed duplicate counts:")
for split in SPLITS:
    print(f"  {split:10s}: {removed_counts[split]:,}")
print(f"  total     : {removed_counts['total']:,}")
print()

print(
    "Verification: "
    + str(
        initial_counts["total"]
        == final_counts["total"] + removed_counts["total"]
    )
)