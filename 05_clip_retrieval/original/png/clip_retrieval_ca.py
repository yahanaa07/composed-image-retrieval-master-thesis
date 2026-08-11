import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from torchmetrics.retrieval import RetrievalRecall

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# The original (Catalan) CSV used only for the ROW ALIGNMENT SANITY CHECK
# below (its row order/count must match cir_db.csv, the file
# build_ground_truth.py used to build row_idx). It is NOT re-read for
# id1/id2/partition — that all comes from the ground-truth files produced by
# build_ground_truth.py. Renamed from TRANSLATED_CSV_PATH since this run's
# CSV is the original/Catalan one, not a translated one — same role, same
# check, just an accurate name.
ORIGINAL_CSV_PATH = Path(
    "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset_updated/original/"
    "cir_db_updated_png.csv"
)

# Sibling of results_clip_retrieval/ (not nested inside it) — must match
# build_ground_truth.py's OUT_DIR exactly.
GROUND_TRUTH_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/original/png/"
    "results_ground_truth_ca"
)

CAPTIONS_WITH_HIER_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/04_caption_generation_with_hierarchies/"
    "original/png/results_captions_original_png"
)
CAPTIONS_WITHOUT_HIER_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/04_caption_generation_without_hierarchies/"
    "original/png/results_captions_original_png"
)

# Base results folder. Each CLIP checkpoint in CLIP_MODELS gets its own
# subfolder underneath, named "<model_name>_<pretrained>" — e.g. this base
# path plus "ViT-H-14_dfn5b/".
CLIP_OUT_DIR_BASE = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/original/png/"
    "results_clip_retrieval_ca"
)

# Image folders — searched in this order regardless of a query's partition
# label (a row's test_id/test_ood tag only concerns which db_processed.txt
# gallery to check id2 against; the actual PNG can live under any split).
# Unchanged from the translated run: the physical PNG files are the same
# images either way — only the captions describing them differ.
IMAGE_DIRS = [
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/test_id"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/test_ood"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/train"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/val"),
]

PARTITIONS = ["test_id", "test_ood"]

# ── Caption JSON files — one per combination ─────────────────────────────────
CAPTION_MODES: dict[str, Path] = {
    "first_first":  CAPTIONS_WITH_HIER_DIR / "descriptions_first_hierarchy.json",
    "full_full":    CAPTIONS_WITH_HIER_DIR / "descriptions_full_hierarchy.json",
    "first_full":   CAPTIONS_WITH_HIER_DIR / "descriptions_mat_first_obj_full.json",
    "full_first":   CAPTIONS_WITH_HIER_DIR / "descriptions_mat_full_obj_first.json",
    "no_hierarchy": CAPTIONS_WITHOUT_HIER_DIR / "descriptions_no_hierarchy.json",
}

# Which of the two caption-generation experiments each mode belongs to —
# used only to group/label metrics_summary.csv.
MODE_CATEGORY: dict[str, str] = {
    "first_first":  "with_hierarchy",
    "full_full":    "with_hierarchy",
    "first_full":   "with_hierarchy",
    "full_first":   "with_hierarchy",
    "no_hierarchy": "without_hierarchy",
}

# Labels for the summary table / console output
MODE_LABELS: dict[str, str] = {
    "first_first":  "mat=FIRST  obj=FIRST",
    "full_full":    "mat=FULL   obj=FULL ",
    "first_full":   "mat=FIRST  obj=FULL ",
    "full_first":   "mat=FULL   obj=FIRST",
    "no_hierarchy": "no hierarchy (target only)",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLIP SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Every (open_clip model architecture, pretrained tag) pair to run, in order.
# Each pair is trained on a different image-text corpus (or a different
# architecture entirely), so results genuinely differ between rows here —
# that's the point of running all four. Listed smallest-first so a config
# problem shows up on the cheapest run before the expensive ones.
CLIP_MODELS: list[tuple[str, str]] = [
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-H-14", "dfn5b"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
]

TEXT_BATCH  = 128    # queries encoded in batches of 128
# Candidate images encoded in batches of 16 — sized for the largest model in
# CLIP_MODELS (ViT-H-14, ~630M params). Smaller models (ViT-B-16, ViT-L-14)
# would comfortably tolerate a bigger batch, but 16 works for all four
# without needing per-model tuning, so it's kept as one shared constant.
# Lower it if you hit a CUDA out-of-memory error; raise it for faster runs
# once you've confirmed your GPU has headroom on ViT-H-14.
IMAGE_BATCH = 16

# Recall@K values to compute
RECALL_KS = [1, 5, 10, 50]

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def find_image(image_id: str) -> Path | None:
    for folder in IMAGE_DIRS:
        p = folder / f"{image_id}.png"
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH LOADING  (produced by build_ground_truth.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(partition: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Loads the (Q, C) binary ground-truth matrix, the length-C gallery id
    list, and the length-Q query metadata table build_ground_truth.py wrote
    for this partition.
    """
    matrix_path = GROUND_TRUTH_DIR / f"gt_matrix_{partition}.npy"
    gallery_path = GROUND_TRUTH_DIR / f"gallery_{partition}.json"
    rows_path = GROUND_TRUTH_DIR / f"queries_{partition}.csv"

    for p in (matrix_path, gallery_path, rows_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ground-truth file: {p}\n"
                f"Run build_ground_truth.py first."
            )

    matrix = np.load(matrix_path)
    with open(gallery_path) as f:
        gallery = json.load(f)
    row_meta = pd.read_csv(rows_path)

    if matrix.shape != (len(row_meta), len(gallery)):
        raise ValueError(
            f"[{partition}] ground-truth shape mismatch: matrix={matrix.shape}, "
            f"expected ({len(row_meta)}, {len(gallery)}) from "
            f"queries_{partition}.csv / gallery_{partition}.json"
        )

    log.info(f"Loaded ground truth [{partition}]: matrix={matrix.shape}, "
             f"{len(gallery)} gallery ids, {len(row_meta)} queries")
    return matrix, gallery, row_meta


# ─────────────────────────────────────────────────────────────────────────────
# ROW ALIGNMENT SANITY CHECK
#
# Caption description caches are keyed by str(row_idx), where row_idx is the
# 0-based row index of cir_db.csv (see build_ground_truth.py). Captions were
# generated from ORIGINAL_CSV_PATH, not cir_db.csv itself. If that file
# doesn't have the exact same id1 values in the exact same row order, the
# description looked up for a given row_idx would silently belong to the
# WRONG query. This check catches that early instead of producing quietly
# wrong retrieval results.
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_row_alignment(row_meta_by_partition: dict[str, pd.DataFrame]) -> None:
    if not ORIGINAL_CSV_PATH.exists():
        log.warning(f"Row-alignment check skipped — {ORIGINAL_CSV_PATH} not found. "
                    f"Proceeding on the assumption that its rows line up 1:1 with "
                    f"cir_db.csv (same order, same count).")
        return

    original_df = pd.read_csv(ORIGINAL_CSV_PATH)
    mismatches = 0
    checked = 0
    for partition, row_meta in row_meta_by_partition.items():
        for _, r in row_meta.iterrows():
            idx = int(r["row_idx"])
            checked += 1
            if idx >= len(original_df):
                mismatches += 1
                continue
            if str(original_df.iloc[idx].get("id1", "")) != str(r["id1"]):
                mismatches += 1

    if mismatches:
        log.error(
            f"ROW ALIGNMENT CHECK FAILED: {mismatches}/{checked} rows have a "
            f"different id1 in {ORIGINAL_CSV_PATH.name} at the same row_idx "
            f"as cir_db.csv. Caption lookups by row_idx would be WRONG for "
            f"these rows. This means the original CSV used for caption "
            f"generation does not have the same row order/count as the CSV "
            f"build_ground_truth.py used. Stop and reconcile the two files "
            f"before trusting these results."
        )
        raise RuntimeError("Row alignment check failed — see log above.")
    log.info(f"Row alignment check passed: {checked} rows verified against "
             f"{ORIGINAL_CSV_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLIP LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(model_name: str, pretrained: str, device: torch.device):
    log.info(f"Loading CLIP: {model_name} / {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    ctx = getattr(model, "context_length", 77)
    log.info(f"CLIP loaded. context_length={ctx} tokens")
    return model, tokenizer, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# BATCH ENCODING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(texts: list[str], model, tokenizer,
                  device: torch.device, batch_size: int = 128,
                  desc: str = "Encoding texts") -> torch.Tensor:
    """
    L2-normalised CLIP text embeddings.

    Returns: float32 tensor of shape (N, D) — kept on CPU.
    Processing in batches satisfies "predictions in matrix, doing in batch".
    """
    all_embs: list[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i: i + batch_size]
        tokens = tokenizer(batch).to(device)   # auto-truncates at 77 tokens
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs, dim=0)           # (N, D)


@torch.no_grad()
def encode_images(candidate_ids: list[str], model, preprocess,
                   device: torch.device, batch_size: int = 16
                   ) -> tuple[torch.Tensor, list[str]]:
    """
    L2-normalised CLIP image embeddings.

    Returns:
      embeddings  — float32 tensor (C, D) on CPU
      valid_ids   — list of candidate IDs that had an image file, IN THE
                    SAME RELATIVE ORDER as candidate_ids (a subsequence) —
                    this ordering guarantee is what align_gallery_columns()
                    below relies on to keep the ground-truth matrix aligned.
    """
    valid_ids: list[str] = []
    valid_paths: list[Path] = []
    for iid in candidate_ids:
        p = find_image(iid)
        if p:
            valid_ids.append(iid)
            valid_paths.append(p)
        else:
            log.warning(f"  No image for candidate '{iid}' — excluded from pool")

    log.info(f"Candidate images found: {len(valid_ids)} / {len(candidate_ids)}")

    all_embs: list[torch.Tensor] = []
    for i in tqdm(range(0, len(valid_paths), batch_size), desc="Encoding candidates"):
        imgs: list[torch.Tensor] = []
        for p in valid_paths[i: i + batch_size]:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                log.warning(f"  Failed to load {p}: {e}")
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(device)
        emb = model.encode_image(batch_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)      # L2 normalise
        all_embs.append(emb.cpu().float())

    if not all_embs:
        emb_dim = model.visual.output_dim if hasattr(model, "visual") else 768
        return torch.zeros((0, emb_dim), dtype=torch.float32), valid_ids

    return torch.cat(all_embs, dim=0), valid_ids          # (C, D)


# ─────────────────────────────────────────────────────────────────────────────
# ALIGN GROUND-TRUTH MATRIX TO THE ACTUAL (ON-DISK) CANDIDATE POOL
# ─────────────────────────────────────────────────────────────────────────────

def align_gallery_columns(gt_matrix: np.ndarray, gallery: list[str],
                           valid_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    encode_images() walks `gallery` in order and drops any id whose PNG isn't
    on disk, returning `valid_ids` — a subsequence of `gallery` in the same
    relative order. Rebuild that same boolean column mask here so gt_matrix
    keeps lining up column-for-column with the candidate embedding matrix.

    Returns
    -------
    gt_matrix_valid : (Q, len(valid_ids)) — gt_matrix with dropped-image
                       columns removed
    gallery_valid   : == valid_ids (returned for a single readable name at
                       call sites)
    """
    valid_set = set(valid_ids)
    mask = np.array([img_id in valid_set for img_id in gallery], dtype=bool)
    assert mask.sum() == len(valid_ids), (
        "gallery/valid_ids mismatch — encode_images() must preserve order "
        "and only drop entries, never reorder or duplicate them."
    )
    return gt_matrix[:, mask], valid_ids


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY MATRIX  (cosine, because embeddings are L2-normalised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_matrix(query_embs: torch.Tensor,
                               candidate_embs: torch.Tensor) -> torch.Tensor:
    """
    Batch cosine-similarity matrix — computed "as usual" (L2-normalise both
    sides, then a single dot product / matmul).

    query_embs     : (Q, D)  — L2-normalised
    candidate_embs : (C, D)  — L2-normalised
    Returns        : (Q, C)  float32 — dot product = cosine similarity
    """
    return query_embs @ candidate_embs.T     # (Q, C)


# ─────────────────────────────────────────────────────────────────────────────
# TORCHMETRICS TENSORS
# ─────────────────────────────────────────────────────────────────────────────

def build_torchmetrics_tensors(
    scores: torch.Tensor,       # (Q, C) float — cosine similarities
    gt_matrix: np.ndarray,      # (Q, C) uint8  — from build_ground_truth.py,
                                 #                already column-aligned to scores
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten the (Q, C) score matrix and the (Q, C) ground-truth matrix into
    the three 1-D tensors required by torchmetrics.retrieval.RetrievalRecall:

      preds   (float)        — cosine similarity score for each (query, candidate) pair
      target  (binary/long)  — 1 if candidate is a ground-truth match, else 0
      indexes (long)         — query index: groups each pair back to its query

    EMPTY GROUND TRUTH
    ───────────────────
    Rows with no ground-truth column set to 1 (all-zero row) produce an
    empty gt set for that query. torchmetrics handles these via
    empty_target_action='neg' (counts as 0 for that query).
    """
    if scores.shape != gt_matrix.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and gt_matrix "
                          f"{gt_matrix.shape} must have the same shape")

    Q, C = scores.shape
    preds = scores.flatten().float()
    target = torch.from_numpy(gt_matrix.astype(np.int64)).flatten()
    indexes = torch.arange(Q).unsqueeze(1).expand(Q, C).flatten().long()

    return preds, target, indexes


# ─────────────────────────────────────────────────────────────────────────────
# RECALL@K via torchmetrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(
    preds: torch.Tensor,     # float (Q*C,)
    target: torch.Tensor,    # long  (Q*C,)  binary 0/1
    indexes: torch.Tensor,   # long  (Q*C,)  query group
    k: int,
) -> float:
    """
    Wrapper around torchmetrics RetrievalRecall.

    empty_target_action='neg': queries with no positive target count as 0.0.
    This matches the convention used in ZS-CIR benchmarks.
    """
    metric = RetrievalRecall(top_k=k, empty_target_action="neg")
    return float(metric(preds, target, indexes=indexes).item())


def compute_all_recalls(
    preds: torch.Tensor,
    target: torch.Tensor,
    indexes: torch.Tensor,
    ks: list[int] = RECALL_KS,
) -> dict[str, float]:
    # *100 -> percentage style (e.g. 32.12 instead of 0.3212), matching how
    # Recall@K is conventionally reported in CIR papers.
    return {
        f"Recall@{k}": round(compute_recall_at_k(preds, target, indexes, k) * 100, 2)
        for k in ks
    }


# ─────────────────────────────────────────────────────────────────────────────
# RANKED RESULTS (for saving to JSON — interpretable output)
# ─────────────────────────────────────────────────────────────────────────────

def build_ranked_results(
    scores: torch.Tensor,        # (Q, C) float
    gt_matrix: np.ndarray,       # (Q, C) uint8, column-aligned to scores
    gallery_valid: list[str],    # length C
    row_meta: pd.DataFrame,      # length Q, same row order as scores
    descriptions: list[str],     # length Q, same row order as scores
    caption_mode: str,
) -> list[dict]:
    """
    Build the per-query result list saved to JSON. Ranking is done on the
    full (Q, C) matrix — no row-by-row Python loops beyond what is needed to
    assemble the output structure.
    """
    ranked_indices = torch.argsort(scores, dim=1, descending=True)  # (Q, C)

    output: list[dict] = []
    for q_idx in range(scores.shape[0]):
        meta = row_meta.iloc[q_idx]
        gt_cols = np.nonzero(gt_matrix[q_idx])[0]
        gt_ids = sorted(gallery_valid[j] for j in gt_cols)

        sorted_cols = ranked_indices[q_idx].tolist()
        output.append({
            "row_idx":       int(meta["row_idx"]),
            "id1":           meta["id1"],
            "id2_raw":       meta["id2_raw"],
            "query":         meta["query"],
            "caption_mode":  caption_mode,
            "description":   descriptions[q_idx],
            "ground_truth":  gt_ids,
            "ranked_ids":    [gallery_valid[j] for j in sorted_cols],
            "ranked_scores": [float(scores[q_idx, j]) for j in sorted_cols],
        })
    return output


# ─────────────────────────────────────────────────────────────────────────────
# ONE RETRIEVAL RUN  (one caption mode x one partition)
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval(
    descriptions: list[str],
    row_meta: pd.DataFrame,
    gt_matrix: np.ndarray,          # (Q, C_valid) already column-aligned
    candidate_embs: torch.Tensor,   # (C_valid, D) L2-normalised
    gallery_valid: list[str],       # length C_valid
    clip_model,
    tokenizer,
    device: torch.device,
    caption_mode: str,
    partition: str,
) -> tuple[list[dict], dict[str, float]]:
    """
    Full retrieval pipeline for one (caption_mode, partition) combination.

    Steps:
      1. Batch-encode query descriptions -> (Q, D) float tensor
      2. Cosine-similarity matrix        -> (Q, C) = query_embs @ cand_embs.T
      3. Build torchmetrics tensors      -> preds / target / indexes,
                                             target comes from gt_matrix
      4. Compute Recall@K via torchmetrics.retrieval.RetrievalRecall
      5. Build ranked results list for JSON output
    """
    # 1. Batch text encoding
    query_embs: torch.Tensor = encode_texts(
        descriptions, clip_model, tokenizer, device, TEXT_BATCH,
        desc=f"Encoding [{caption_mode}] queries — {partition}"
    )  # (Q, D)

    # 2. Full cosine-similarity matrix — one matmul ("in batch")
    scores: torch.Tensor = compute_similarity_matrix(
        query_embs, candidate_embs
    )  # (Q, C)

    # 3. Flatten into torchmetrics format, target from the prebuilt gt matrix
    preds, target, indexes = build_torchmetrics_tensors(scores, gt_matrix)

    # 4. Recall@K via torchmetrics RetrievalRecall
    metrics = compute_all_recalls(preds, target, indexes)

    # 5. Ranked output for JSON
    ranked_results = build_ranked_results(
        scores, gt_matrix, gallery_valid, row_meta, descriptions, caption_mode
    )

    return ranked_results, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ONE CHECKPOINT, ALL PARTITIONS x ALL CAPTION MODES
# ─────────────────────────────────────────────────────────────────────────────

def run_for_model(
    model_name: str,
    pretrained: str,
    gt_data: dict[str, tuple[np.ndarray, list[str], pd.DataFrame]],
    caches: dict[str, dict[str, str]],
    device: torch.device,
) -> dict[str, dict]:
    """
    Runs the full retrieval pipeline (both partitions x all 5 caption modes)
    for one CLIP checkpoint, saving everything under its own subfolder.

    Returns the {"{mode}_{partition}": {...}} metrics dict for this
    checkpoint (used by main() to build the cross-model comparison file).
    """
    run_label = f"{model_name}_{pretrained}"
    out_dir = CLIP_OUT_DIR_BASE / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'#'*90}")
    log.info(f"# MODEL: {run_label}")
    log.info(f"{'#'*90}")
    log.info(f"Output folder: {out_dir}")

    clip_model, tokenizer, preprocess = load_clip(model_name, pretrained, device)

    all_metrics: dict[str, dict] = {}

    for partition in PARTITIONS:
        log.info(f"\n{'='*65}")
        log.info(f"[{run_label}] Partition: {partition}")

        gt_matrix, gallery, row_meta = gt_data[partition]
        log.info(f"Queries: {len(row_meta)}  |  Gallery (from ground truth): {len(gallery)}")

        # Encode candidates ONCE per (model, partition) — shared across all
        # five caption modes. Must be redone for every model: each
        # checkpoint has its own preprocessing transform and embedding space.
        cand_embs, cand_valid = encode_images(
            gallery, clip_model, preprocess, device, IMAGE_BATCH
        )  # (C_valid, D)
        log.info(f"Candidate embeddings: {cand_embs.shape}  "
                 f"({len(cand_valid)} / {len(gallery)} images found on disk)")

        if cand_embs.shape[0] == 0:
            log.error(f"No candidate images found for {partition} — skipping")
            continue

        # Drop the same columns from the ground-truth matrix so it stays
        # aligned with the candidate embeddings.
        gt_matrix_valid, gallery_valid = align_gallery_columns(gt_matrix, gallery, cand_valid)

        # Sanity check: how many queries still have >=1 ground-truth target
        # after dropping images missing on disk.
        covered = int((gt_matrix_valid.sum(axis=1) > 0).sum())
        log.info(f"Queries with >=1 ground-truth target present after "
                 f"disk-availability filtering: {covered} / {len(row_meta)}")

        # Run retrieval for each of the five caption modes
        for mode in CAPTION_MODES:
            label = MODE_LABELS[mode]
            log.info(f"\n  [{run_label}] Run [{label}] — {partition}")

            descriptions = [caches[mode].get(str(idx), "") for idx in row_meta["row_idx"]]
            n_empty = sum(1 for d in descriptions if not d)
            if n_empty:
                log.warning(f"  [{mode}/{partition}] {n_empty}/{len(descriptions)} "
                            f"rows have no cached description (missing from JSON) "
                            f"— they will be encoded as empty strings")

            results, metrics = run_retrieval(
                descriptions, row_meta, gt_matrix_valid,
                cand_embs, gallery_valid,
                clip_model, tokenizer, device,
                mode, partition,
            )

            log.info(f"  Metrics [{mode}/{partition}]: {metrics}")
            all_metrics[f"{mode}_{partition}"] = {
                "caption_mode": mode,
                "category": MODE_CATEGORY[mode],
                "partition": partition,
                "num_queries": len(row_meta),
                "gallery_size": len(gallery_valid),
                **metrics,
            }

            out_path = out_dir / f"results_{mode}_{partition}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log.info(f"  Saved -> {out_path}")

    # ── Save this checkpoint's metrics summary (JSON, recall-only + CSV, detailed) ──
    recall_only = {
        key: {k: v for k, v in mets.items() if k.startswith("Recall@")}
        for key, mets in all_metrics.items()
    }
    summary_json_path = out_dir / "metrics_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(recall_only, f, indent=2)
    log.info(f"\nMetrics summary (json) -> {summary_json_path}")

    summary_df = pd.DataFrame(all_metrics.values())
    summary_csv_path = out_dir / "metrics_summary_detailed.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    log.info(f"Metrics summary (csv)  -> {summary_csv_path}")

    # ── Console table for this checkpoint ──────────────────────────────────
    print("\n" + "=" * 90)
    print(f"RETRIEVAL RESULTS — {run_label} — RECALL@K (torchmetrics RetrievalRecall)")
    print("=" * 90)
    print(f"{'Caption combination':<28}  {'Partition':<10}  "
          f"{'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'R@50':>7}")
    print("─" * 90)
    for key, mets in all_metrics.items():
        mode = mets["caption_mode"]
        part = mets["partition"]
        label = MODE_LABELS.get(mode, mode)
        print(f"  {label:<26}  {part:<10}  "
              f"{mets.get('Recall@1',  0):>7.2f}  "
              f"{mets.get('Recall@5',  0):>7.2f}  "
              f"{mets.get('Recall@10', 0):>7.2f}  "
              f"{mets.get('Recall@50', 0):>7.2f}")
    print()
    print(f"All result files for {run_label} saved to: {out_dir}")
    print()

    # ── Sample analysis (uses full_full test_id as reference, if available) ──
    _print_sample_analysis(out_dir)

    # ── Free this checkpoint before loading the next one ──────────────────
    del clip_model, tokenizer, preprocess
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_metrics


def _print_sample_analysis(out_dir: Path) -> None:
    """Print a quick sanity-check over the first 2 rows of one result file."""
    sample_path = out_dir / "results_full_full_test_id.json"
    sample_save_path = out_dir / "sample_analysis.json"

    if not sample_path.exists():
        log.info("Sample file not found — skipping sample analysis")
        return

    with open(sample_path) as f:
        res: list[dict] = json.load(f)

    print("SAMPLE (first 2 queries, full_full captions, test_id):")
    print("-" * 80)
    sample_analysis: list[dict] = []

    for r in res[:2]:
        gt = set(r["ground_truth"])
        ranked = r["ranked_ids"]

        first_correct_rank = next(
            (i + 1 for i, x in enumerate(ranked) if x in gt), None
        )
        hit_dict = {f"hit@{k}": bool(gt & set(ranked[:k])) for k in RECALL_KS}

        print(f"  id1     : {r['id1']}")
        print(f"  query   : {r['query']}")
        print(f"  gt      : {r['ground_truth'][:3]}")
        print(f"  top-3   : {ranked[:3]}")
        for k in RECALL_KS:
            symbol = "YES" if hit_dict[f"hit@{k}"] else "no "
            print(f"  hit@{k:<2}: {symbol}")
        print(f"  first correct rank: {first_correct_rank}")
        print("-" * 80)

        sample_analysis.append({
            "id1":                r["id1"],
            "query":               r["query"],
            "ground_truth":        r["ground_truth"],
            "top3":                ranked[:3],
            "hit_metrics":         hit_dict,
            "first_correct_rank":  first_correct_rank,
        })

    with open(sample_save_path, "w") as f:
        json.dump(sample_analysis, f, indent=2)
    print(f"\nSaved sample analysis -> {sample_save_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    CLIP_OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    # ── Load ground truth for both partitions ONCE (shared by every model) ──
    gt_data: dict[str, tuple[np.ndarray, list[str], pd.DataFrame]] = {}
    for partition in PARTITIONS:
        gt_data[partition] = load_ground_truth(partition)

    # ── Row-alignment sanity check before touching CLIP at all ────────────
    sanity_check_row_alignment({p: gt_data[p][2] for p in PARTITIONS})

    # ── Load all five caption caches ONCE (shared by every model) ─────────
    missing_files = [p for p in CAPTION_MODES.values() if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Caption file(s) not found:\n"
            + "\n".join(f"  {p}" for p in missing_files)
            + "\nRun the caption-generation scripts first (4 'with hierarchy' "
              "scripts + the 'no hierarchy' script)."
        )

    caches: dict[str, dict[str, str]] = {}
    for mode, path in CAPTION_MODES.items():
        with open(path) as f:
            caches[mode] = json.load(f)
        log.info(f"Loaded [{mode}]: {len(caches[mode])} descriptions from {path.name}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info(f"  GPU: {p.name}  {p.total_memory/1e9:.1f} GB")

    # ── Run every checkpoint in CLIP_MODELS, one at a time ─────────────────
    metrics_by_model: dict[str, dict[str, dict]] = {}
    for model_name, pretrained in CLIP_MODELS:
        run_label = f"{model_name}_{pretrained}"
        metrics_by_model[run_label] = run_for_model(
            model_name, pretrained, gt_data, caches, device
        )

    # ── Cross-model comparison file (top-level, not inside any model folder) ──
    rows = []
    for run_label, all_metrics in metrics_by_model.items():
        for key, mets in all_metrics.items():
            rows.append({"model": run_label, **mets})

    combined_df = pd.DataFrame(rows)
    combined_csv_path = CLIP_OUT_DIR_BASE / "all_models_metrics_summary.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    log.info(f"\nCross-model summary (csv)  -> {combined_csv_path}")

    combined_json_path = CLIP_OUT_DIR_BASE / "all_models_metrics_summary.json"
    with open(combined_json_path, "w") as f:
        json.dump(metrics_by_model, f, indent=2)
    log.info(f"Cross-model summary (json) -> {combined_json_path}")

    # ── Final grand comparison table ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("ALL MODELS — RECALL@K COMPARISON")
    print("=" * 100)
    print(f"{'Model':<28}  {'Caption combination':<24}  {'Partition':<10}  "
          f"{'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'R@50':>7}")
    print("─" * 100)
    for _, row in combined_df.iterrows():
        label = MODE_LABELS.get(row["caption_mode"], row["caption_mode"])
        print(f"  {row['model']:<26}  {label:<22}  {row['partition']:<10}  "
              f"{row.get('Recall@1',  0):>7.2f}  "
              f"{row.get('Recall@5',  0):>7.2f}  "
              f"{row.get('Recall@10', 0):>7.2f}  "
              f"{row.get('Recall@50', 0):>7.2f}")
    print()
    print(f"Per-model result files under: {CLIP_OUT_DIR_BASE}/<model_name>_<pretrained>/")
    print(f"Cross-model summary at      : {combined_csv_path}")
    print()


if __name__ == "__main__":
    main()