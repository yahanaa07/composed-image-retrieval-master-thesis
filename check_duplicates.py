"""
Duplicate detection for downloaded EUFCC-340K images.

Checks each split folder (test_id, test_ood, val, train) under
newly_downloaded_images_png in two ways:

1. ID-based duplicates: groups filenames by a normalized version of the id
   (stripped whitespace, case-insensitive) so that things like '123.png' vs
   '123 .png' or 'ABC.png' vs 'abc.png' are caught, even if the raw filenames
   differ slightly.

2. Content-based duplicates: opens each image, converts to RGB, and hashes
   the raw pixel bytes (md5). Images with identical pixel content end up in
   the same hash bucket regardless of what their filename/id is. This is the
   "check the content of the id, not the id itself" check.

CPU only - no GPU is used or required anywhere in this script.

Results are written as JSON:
  - id_only/id_duplicates.json
  - content_itself/content_duplicates.json

Both result folders are created automatically if they don't already exist.
"""

import os
import json
import hashlib
from collections import defaultdict
from datetime import datetime
import concurrent.futures

from PIL import Image
import PIL
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGES_ROOT = (
    r"C:\YAHANAA\OneDrive\Thesis project\03_finding_missing_images_png"
    r"\newly_downloaded_images_png"
)

ID_RESULTS_DIR = r"C:\YAHANAA\OneDrive\Thesis project\02_repetition_check\results\id_only"
CONTENT_RESULTS_DIR = r"C:\YAHANAA\OneDrive\Thesis project\02_repetition_check\results\content_itself"

SPLITS = ["test_id", "test_ood", "val", "train"]

NUM_THREADS = 12  # only used to parallelize file I/O + hashing, still CPU-only


def ensure_dirs():
    os.makedirs(ID_RESULTS_DIR, exist_ok=True)
    os.makedirs(CONTENT_RESULTS_DIR, exist_ok=True)


def list_images(split_folder):
    """Return list of (id, full_path) for every .png in a split folder."""
    items = []
    if not os.path.isdir(split_folder):
        print(f"  WARNING: folder not found: {split_folder}")
        return items
    for fname in os.listdir(split_folder):
        if fname.lower().endswith(".png"):
            img_id = os.path.splitext(fname)[0]
            items.append((img_id, os.path.join(split_folder, fname)))
    return items


# ---------------------------------------------------------------------------
# 1. Duplicate check by ID (filename-based)
# ---------------------------------------------------------------------------
def find_id_duplicates(items):
    """
    Groups filenames by a normalized id (case-insensitive, whitespace
    stripped). Returns dict: normalized_id -> list of original ids that
    collide under that normalization.
    """
    buckets = defaultdict(list)
    for img_id, _ in items:
        norm = img_id.strip().lower()
        buckets[norm].append(img_id)

    duplicates = {norm: ids for norm, ids in buckets.items() if len(ids) > 1}
    return duplicates


# ---------------------------------------------------------------------------
# 2. Duplicate check by content (pixel hash)
# ---------------------------------------------------------------------------
def hash_image_content(item):
    img_id, path = item
    try:
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            digest = hashlib.md5(im.tobytes()).hexdigest()
        return img_id, digest, None
    except (OSError, PIL.UnidentifiedImageError) as e:
        return img_id, None, str(e)


def find_content_duplicates(items, split, num_threads=NUM_THREADS):
    hash_to_ids = defaultdict(list)
    errors = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(hash_image_content, item) for item in items]
        for future in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc=f"hashing {split}"):
            img_id, digest, err = future.result()
            if err is not None:
                errors[img_id] = err
                continue
            hash_to_ids[digest].append(img_id)

    duplicates = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    return duplicates, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    id_report = {"generated_at": datetime.now().isoformat(), "splits": {}}
    content_report = {"generated_at": datetime.now().isoformat(), "splits": {}}

    for split in SPLITS:
        split_folder = os.path.join(IMAGES_ROOT, split)
        items = list_images(split_folder)
        print(f"\n=== {split}: {len(items)} images found ===")

        if not items:
            id_report["splits"][split] = {
                "folder": split_folder,
                "total_images": 0,
                "num_duplicate_groups": 0,
                "num_duplicate_images": 0,
                "duplicates": {},
            }
            content_report["splits"][split] = {
                "folder": split_folder,
                "total_images": 0,
                "num_duplicate_groups": 0,
                "num_duplicate_images": 0,
                "duplicates": {},
                "unreadable_files": {},
            }
            continue

        # --- 1. id-based ---
        id_dupes = find_id_duplicates(items)
        num_dup_images_id = sum(len(v) for v in id_dupes.values())
        id_report["splits"][split] = {
            "folder": split_folder,
            "total_images": len(items),
            "num_duplicate_groups": len(id_dupes),
            "num_duplicate_images": num_dup_images_id,
            "duplicates": id_dupes,
        }
        print(f"  id-based duplicates: {len(id_dupes)} groups, {num_dup_images_id} images involved")

        # --- 2. content-based ---
        content_dupes, errors = find_content_duplicates(items, split)
        num_dup_images_content = sum(len(v) for v in content_dupes.values())
        content_report["splits"][split] = {
            "folder": split_folder,
            "total_images": len(items),
            "num_duplicate_groups": len(content_dupes),
            "num_duplicate_images": num_dup_images_content,
            "duplicates": content_dupes,
            "unreadable_files": errors,
        }
        print(f"  content-based duplicates: {len(content_dupes)} groups, {num_dup_images_content} images involved")
        if errors:
            print(f"  {len(errors)} files could not be opened/hashed")

    # --- save reports ---
    id_out_path = os.path.join(ID_RESULTS_DIR, "id_duplicates.json")
    content_out_path = os.path.join(CONTENT_RESULTS_DIR, "content_duplicates.json")

    with open(id_out_path, "w", encoding="utf-8") as f:
        json.dump(id_report, f, indent=2)

    with open(content_out_path, "w", encoding="utf-8") as f:
        json.dump(content_report, f, indent=2)

    print(f"\nSaved id-based duplicate report to: {id_out_path}")
    print(f"Saved content-based duplicate report to: {content_out_path}")


if __name__ == "__main__":
    main()