import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = Path(
    "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset_updated/translated/png/"
    "cir_db_updated_png_translated.csv"
)
# Unchanged — still the correct manifest for the updated CSV above: every
# id1/id2 referenced there was checked and is fully covered by this file
# (100% of id2 targets and id1 references were found in it).
DB_PROCESSED_PATH = Path(
    "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset/original/db_processed.txt"
)
# Sibling of results_clip_retrieval/ (not nested inside it) — clip_retrieval.py's
# GROUND_TRUTH_DIR must point at this exact same folder.
OUT_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/translated/png/"
    "results_ground_truth"
)

# cir_db.csv only ever labels rows test_id / test_ood. db_processed.txt also
# contains train / val / test_human images, which are irrelevant here since
# no cir_db.csv row ever needs to be checked against them.
PARTITIONS = ["test_id", "test_ood"]

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_id2(raw) -> list[str]:
    """
    Split a (possibly multi-valued, comma-separated) id2 cell into a clean
    list of individual target image ids. Handles NaN / empty cells.

    "ceramic_011219"                                  -> ["ceramic_011219"]
    "art_166248, art_166267, art_010762, art_166586"  -> ["art_166248", "art_166267", "art_010762", "art_166586"]
    NaN / ""                                          -> []
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p and p.lower() != "nan"]


def load_galleries(db_processed_path: Path) -> dict[str, list[str]]:
    """
    Step (i): read db_processed.txt (image_id, partition — no header),
    keep only the test_id / test_ood rows, dedupe per partition, and sort
    for a stable column order.
    """
    log.info(f"Loading gallery manifest: {db_processed_path}")
    proc = pd.read_csv(db_processed_path, header=None, names=["image_id", "partition"])
    proc["image_id"] = proc["image_id"].astype(str).str.strip()
    proc["partition"] = proc["partition"].astype(str).str.strip()

    log.info(f"  db_processed.txt partitions found: "
             f"{proc['partition'].value_counts().to_dict()}")

    galleries: dict[str, list[str]] = {}
    for part in PARTITIONS:
        ids = sorted(set(proc.loc[proc["partition"] == part, "image_id"]))
        galleries[part] = ids
        log.info(f"  Gallery [{part}]: {len(ids)} unique image ids "
                 f"(sorted order = matrix column order)")
    return galleries


def build_partition_matrix(part_df: pd.DataFrame,
                            gallery: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Steps (ii)-(iv): for every row (query) in this partition, mark a 1 in
    every gallery column whose image id is one of that row's id2 targets,
    0 everywhere else.

    Returns
    -------
    matrix   : (num_queries, gallery_size) uint8 binary numpy array —
               row q, col j == 1 iff gallery[j] is a ground-truth target
               of query q.
    row_meta : DataFrame with one row per query, in the same order as the
               matrix rows — row_idx, id1, id2_raw, query, targets_total,
               targets_found_in_gallery.
    """
    col_index = {img_id: j for j, img_id in enumerate(gallery)}
    n_rows, n_cols = len(part_df), len(gallery)
    matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)

    meta_rows = []
    for i, (csv_idx, row) in enumerate(part_df.iterrows()):
        targets = parse_id2(row.get("id2", ""))
        found = [t for t in targets if t in col_index]
        for t in found:
            matrix[i, col_index[t]] = 1
        meta_rows.append({
            "row_idx": csv_idx,              # 0-based row index into CSV_PATH
            "id1": row.get("id1", ""),
            "id2_raw": row.get("id2", ""),
            "query": row.get("query", ""),
            "targets_total": len(targets),
            "targets_found_in_gallery": len(found),
        })

    row_meta = pd.DataFrame(meta_rows)
    return matrix, row_meta


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    log.info(f"Rows: {len(df)} | partitions: {df['partition'].value_counts().to_dict()}")

    unexpected = set(df["partition"].unique()) - set(PARTITIONS)
    if unexpected:
        log.warning(f"cir_db.csv has partition value(s) not in {PARTITIONS}: "
                    f"{unexpected} — those rows will be skipped entirely")

    galleries = load_galleries(DB_PROCESSED_PATH)

    for part in PARTITIONS:
        part_df = df[df["partition"] == part]
        gallery = galleries.get(part, [])

        if part_df.empty:
            log.warning(f"No cir_db.csv rows for partition '{part}' — skipping")
            continue
        if not gallery:
            log.warning(f"No db_processed.txt images for partition '{part}' — skipping")
            continue

        log.info(f"\n{'='*65}")
        log.info(f"Partition: {part}")
        log.info(f"Queries: {len(part_df)}  |  Gallery size: {len(gallery)}")

        matrix, row_meta = build_partition_matrix(part_df, gallery)

        # Step (iii) diagnostic: how many queries had none / some / all of
        # their id2 targets actually present in the gallery.
        none_found = int((row_meta["targets_found_in_gallery"] == 0).sum())
        some_found = int(((row_meta["targets_found_in_gallery"] > 0) &
                           (row_meta["targets_found_in_gallery"] <
                            row_meta["targets_total"])).sum())
        all_found = int(((row_meta["targets_total"] > 0) &
                          (row_meta["targets_found_in_gallery"] ==
                           row_meta["targets_total"])).sum())
        row_sums = matrix.sum(axis=1)
        log.info(f"  Queries with 0 targets found in gallery   : {none_found}")
        log.info(f"  Queries with SOME (not all) targets found : {some_found}")
        log.info(f"  Queries with ALL targets found             : {all_found}")
        log.info(f"  Ground-truth hits per query — min:{row_sums.min()}  "
                 f"max:{row_sums.max()}  mean:{row_sums.mean():.3f}")
        log.info(f"  Matrix shape: {matrix.shape}  |  total positives: {int(matrix.sum())}")

        if none_found > 0:
            log.warning(f"  {none_found} quer{'y' if none_found == 1 else 'ies'} in "
                        f"'{part}' have NO ground-truth image in the gallery at all — "
                        f"they will contribute 0 to Recall@K (torchmetrics "
                        f"empty_target_action='neg').")

        # ── save ──────────────────────────────────────────────────────────
        matrix_path = OUT_DIR / f"gt_matrix_{part}.npy"
        gallery_path = OUT_DIR / f"gallery_{part}.json"
        rows_path = OUT_DIR / f"queries_{part}.csv"

        np.save(matrix_path, matrix)
        with open(gallery_path, "w") as f:
            json.dump(gallery, f, indent=2)
        row_meta.to_csv(rows_path, index=False)

        log.info(f"  Saved matrix  -> {matrix_path}  shape={matrix.shape}")
        log.info(f"  Saved gallery -> {gallery_path}")
        log.info(f"  Saved rows    -> {rows_path}")

    log.info("\nGround-truth build complete.")
    log.info(f"All files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()