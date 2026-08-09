import os
import json
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

# Original image location
IMAGES_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "01_download_images/png/downloaded_images_png"
)

# JSON report location in HOME
RESULT_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "02_preprocessing/duplicate_images/"
    "results_duplicates_png"
)

# Folder containing the duplicates already moved
REMOVED_DUPLICATES_ROOT = (
    "/scratch/users/ymathansingh/"
    "duplicate_images/results_duplicates_png/"
    "removed_duplicates"
)

SPLITS = [
    "train",
    "val",
    "test_id",
    "test_ood"
]


# ============================================================
# COUNT PNG FILES
# ============================================================

def count_images(root):
    counts = {}

    for split in SPLITS:
        folder = os.path.join(root, split)

        if not os.path.isdir(folder):
            counts[split] = 0
            continue

        counts[split] = sum(
            1
            for filename in os.listdir(folder)
            if filename.lower().endswith(".png")
        )

    counts["total"] = sum(
        counts[split]
        for split in SPLITS
    )

    return counts


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Count images currently remaining in original location
    # --------------------------------------------------------

    final_counts = count_images(
        IMAGES_ROOT
    )

    # --------------------------------------------------------
    # Count images that were moved to removed_duplicates
    # --------------------------------------------------------

    removed_counts = count_images(
        REMOVED_DUPLICATES_ROOT
    )

    # --------------------------------------------------------
    # Reconstruct original counts
    #
    # Original = remaining + moved
    # --------------------------------------------------------

    reconstructed_initial_counts = {}

    for split in SPLITS:
        reconstructed_initial_counts[split] = (
            final_counts[split]
            + removed_counts[split]
        )

    reconstructed_initial_counts["total"] = sum(
        reconstructed_initial_counts[split]
        for split in SPLITS
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = {
        "report_type": "recovered_after_duplicate_removal",

        "generated_at": datetime.now().isoformat(),

        "paths": {
            "original_images": IMAGES_ROOT,
            "removed_duplicates": REMOVED_DUPLICATES_ROOT,
            "json_report": RESULT_ROOT
        },

        "initial_image_counts_reconstructed": (
            reconstructed_initial_counts
        ),

        "final_image_counts_in_original_location": (
            final_counts
        ),

        "removed_duplicate_counts": (
            removed_counts
        ),

        "duplicate_removal_summary": {
            "images_removed_total": (
                removed_counts["total"]
            ),

            "images_remaining_total": (
                final_counts["total"]
            ),

            "images_before_removal_total": (
                reconstructed_initial_counts["total"]
            ),

            "verification": {
                "before_equals_remaining_plus_removed": (
                    reconstructed_initial_counts["total"]
                    ==
                    final_counts["total"]
                    +
                    removed_counts["total"]
                )
            }
        },

        "note": (
            "This report was reconstructed from the "
            "filesystem after the original duplicate-analysis "
            "job completed its file-moving operation but "
            "failed while writing its JSON report. "
            "No duplicate detection or image moving was "
            "performed by this recovery script. "
            "Therefore, this report contains reliable "
            "filesystem counts but does not reconstruct "
            "the original MD5 duplicate-group details."
        )
    }

    # --------------------------------------------------------
    # Create result directory
    # --------------------------------------------------------

    os.makedirs(
        RESULT_ROOT,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save JSON
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
    # Print results
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Duplicate report recovered successfully")
    print("============================================")

    print()
    print("INITIAL / RECONSTRUCTED")
    print("--------------------------------------------")

    for split in SPLITS:
        print(
            f"{split:10s}: "
            f"{reconstructed_initial_counts[split]:,}"
        )

    print(
        f"{'total':10s}: "
        f"{reconstructed_initial_counts['total']:,}"
    )

    print()
    print("CURRENT ORIGINAL LOCATION")
    print("--------------------------------------------")

    for split in SPLITS:
        print(
            f"{split:10s}: "
            f"{final_counts[split]:,}"
        )

    print(
        f"{'total':10s}: "
        f"{final_counts['total']:,}"
    )

    print()
    print("MOVED DUPLICATES")
    print("--------------------------------------------")

    for split in SPLITS:
        print(
            f"{split:10s}: "
            f"{removed_counts[split]:,}"
        )

    print(
        f"{'total':10s}: "
        f"{removed_counts['total']:,}"
    )

    print()
    print("VERIFICATION")
    print("--------------------------------------------")

    print(
        f"{reconstructed_initial_counts['total']:,} "
        f"= "
        f"{final_counts['total']:,} "
        f"+ "
        f"{removed_counts['total']:,}"
    )

    print(
        "Check:",
        reconstructed_initial_counts["total"]
        ==
        final_counts["total"]
        +
        removed_counts["total"]
    )

    print()
    print("JSON saved to:")
    print(output_file)

    print()
    print("============================================")


if __name__ == "__main__":
    main()
