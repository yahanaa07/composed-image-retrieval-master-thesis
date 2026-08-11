import csv
import json
from pathlib import Path

# Output CSV is always written into results_gallery_trace/, next to this
# script file (wherever that is) — NOT into GROUND_TRUTH_DIR (which only
# holds the SOURCE gallery_*.json files this script reads, and shouldn't be
# cluttered with this script's own output).
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results_gallery_trace"

# Points at the Catalan (original/png/) results_ground_truth_ca/ folder here
# — note the "_ca" suffix your actual folder names use for the Catalan
# variant (results_ground_truth_ca / results_clip_retrieval_ca /
# results_mixture_retrieval_ca), unlike the English/translated variant which
# has no suffix (results_ground_truth / results_clip_retrieval / ...).
#
# Per the note above, this is purely a folder-naming detail — the trace
# itself is identical either way, since gallery_test_id.json /
# gallery_test_ood.json only depend on the shared db_processed.txt, not on
# CSV_PATH. Swap this to "translated/png/results_ground_truth" (no "_ca") if
# you'd rather run it from the English side instead; it makes no difference
# to the result.
GROUND_TRUTH_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/original/png/"
    "results_ground_truth_ca"
)

# The same four folders clip_retrieval.py's find_image() searches.
IMAGES_ROOT = Path(
    "/home/users/ymathansingh/Master_Thesis/01_download_images/png/"
    "downloaded_images_png"
)

# Where check_duplicates.py moved detected duplicates. Split structure is
# preserved: removed_duplicates/train|val|test_id|test_ood/<id>.png
REMOVED_DUPLICATES_ROOT = Path(
    "/scratch/users/ymathansingh/duplicate_images/"
    "results_duplicates_png/removed_duplicates"
)

# Where check_missing.py moved corrupt images. Same split structure:
# corrupt_images_png/train|val|test_id|test_ood/<id>.png
CORRUPT_IMAGES_ROOT = Path(
    "/scratch/users/ymathansingh/missing_images/corrupt_images_png"
)

SPLITS = ["train", "val", "test_id", "test_ood"]
PARTITIONS = ["test_id", "test_ood"]  # the two galleries build_ground_truth.py builds


def find_in(root: Path, image_id: str) -> str | None:
    """Return the split name if {image_id}.png exists under root/<split>/, else None."""
    for split in SPLITS:
        if (root / split / f"{image_id}.png").exists():
            return split
    return None


def main():
    all_rows = []
    print(f"{'Partition':<10} {'Gallery size':>13} {'Found live':>11} "
          f"{'In removed_dup':>15} {'In corrupt':>11} {'Unaccounted':>12}")
    print("-" * 76)

    for part in PARTITIONS:
        gallery_path = GROUND_TRUTH_DIR / f"gallery_{part}.json"
        if not gallery_path.exists():
            print(f"  Skipping {part} — {gallery_path} not found")
            continue

        with open(gallery_path) as f:
            gallery = json.load(f)

        found_live = 0
        in_removed_dup = 0
        in_corrupt = 0
        unaccounted = 0

        for image_id in gallery:
            live_split = find_in(IMAGES_ROOT, image_id)
            if live_split:
                found_live += 1
                continue

            dup_split = find_in(REMOVED_DUPLICATES_ROOT, image_id)
            corrupt_split = find_in(CORRUPT_IMAGES_ROOT, image_id)

            if dup_split:
                in_removed_dup += 1
                status = "found_in_removed_duplicates"
            elif corrupt_split:
                in_corrupt += 1
                status = "found_in_corrupt_images"
            else:
                unaccounted += 1
                status = "unaccounted_for"

            all_rows.append({
                "partition": part,
                "image_id": image_id,
                "found_in_removed_duplicates": dup_split or "",
                "found_in_corrupt_images": corrupt_split or "",
                "status": status,
            })

        print(f"{part:<10} {len(gallery):>13} {found_live:>11} "
              f"{in_removed_dup:>15} {in_corrupt:>11} {unaccounted:>12}")

    if all_rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUTPUT_DIR / "missing_gallery_trace.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved -> {out_csv}  ({len(all_rows)} missing-id rows)")
    else:
        print("\nNo missing gallery ids in any partition — nothing to save.")


if __name__ == "__main__":
    main()