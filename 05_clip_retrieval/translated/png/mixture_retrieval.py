import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import clip_retrieval as cr

# Sibling of results_ground_truth/ and results_clip_retrieval/ (not nested in
# either) — same convention build_ground_truth.py's OUT_DIR already uses.
MIXTURE_OUT_DIR_BASE = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/translated/png/"
    "results_mixture_retrieval"
)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MIXTURE QUERY EMBEDDINGS  (image + text, summed, normalised once)
# ─────────────────────────────────────────────────────────────────────────────

def build_mixture_query_embeddings(
    row_meta: pd.DataFrame,
    clip_model, tokenizer, preprocess,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    For every row in row_meta: load id1's image, CLIP-encode it (RAW), CLIP-
    encode row_meta['query'].strip().replace('for','to') (RAW), sum the two
    raw vectors, L2-normalise ONCE — matches feature_extractor.py's
    `combined` baseline exactly.

    Rows whose id1 image isn't found on disk are dropped (can't build a
    mixture query without the reference image).

    Returns
    -------
    query_embs : (N_valid, D) float32 CPU tensor, in the same relative order
                 as row_meta rows where valid_mask is True.
    valid_mask : (len(row_meta),) bool array — True where the row was kept.
    """
    valid_mask = []
    texts: list[str] = []
    images: list[Image.Image] = []

    for _, r in row_meta.iterrows():
        p = cr.find_image(str(r["id1"]))
        if p is None:
            valid_mask.append(False)
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log.warning(f"  Failed to load id1 image {p}: {e}")
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        images.append(img)
        texts.append(str(r["query"]).strip().replace("for", "to"))

    valid_mask = np.array(valid_mask, dtype=bool)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        log.warning(f"  {n_dropped}/{len(row_meta)} rows dropped from mixture mode "
                    f"— id1 image not found on disk")

    batch_size = cr.IMAGE_BATCH
    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding mixture queries"):
            img_batch = images[i: i + batch_size]
            txt_batch = texts[i: i + batch_size]

            img_tensors = torch.stack([preprocess(im) for im in img_batch]).to(device)
            img_feat = clip_model.encode_image(img_tensors).float()   # RAW, unnormalised

            tokens = tokenizer(txt_batch).to(device)
            txt_feat = clip_model.encode_text(tokens).float()         # RAW, unnormalised

            combined = img_feat + txt_feat
            combined = combined / combined.norm(dim=-1, keepdim=True)  # normalise ONCE
            embs.append(combined.cpu())

    if embs:
        query_embs = torch.cat(embs, dim=0)
    else:
        dim = clip_model.visual.output_dim if hasattr(clip_model, "visual") else 768
        query_embs = torch.zeros((0, dim), dtype=torch.float32)

    return query_embs, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# ONE CHECKPOINT, BOTH PARTITIONS, MIXTURE MODE ONLY
# ─────────────────────────────────────────────────────────────────────────────

def run_mixture_for_model(
    model_name: str,
    pretrained: str,
    gt_data: dict[str, tuple[np.ndarray, list[str], pd.DataFrame]],
    device: torch.device,
) -> dict[str, dict]:
    run_label = f"{model_name}_{pretrained}"
    out_dir = MIXTURE_OUT_DIR_BASE / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'#'*90}")
    log.info(f"# MIXTURE MODE — MODEL: {run_label}")
    log.info(f"{'#'*90}")

    clip_model, tokenizer, preprocess = cr.load_clip(model_name, pretrained, device)

    mixture_metrics: dict[str, dict] = {}

    for partition in cr.PARTITIONS:
        log.info(f"\n{'='*65}")
        log.info(f"[{run_label}] Mixture — Partition: {partition}")

        gt_matrix, gallery, row_meta = gt_data[partition]

        cand_embs, cand_valid = cr.encode_images(
            gallery, clip_model, preprocess, device, cr.IMAGE_BATCH
        )
        if cand_embs.shape[0] == 0:
            log.error(f"No candidate images for {partition} — skipping")
            continue
        gt_matrix_valid, gallery_valid = cr.align_gallery_columns(gt_matrix, gallery, cand_valid)

        query_embs, valid_mask = build_mixture_query_embeddings(
            row_meta, clip_model, tokenizer, preprocess, device
        )
        row_meta_valid = row_meta[valid_mask].reset_index(drop=True)
        gt_matrix_valid2 = gt_matrix_valid[valid_mask]

        if query_embs.shape[0] == 0:
            log.error(f"No valid mixture queries for {partition} — skipping")
            continue

        scores = cr.compute_similarity_matrix(query_embs, cand_embs)
        preds, target, indexes = cr.build_torchmetrics_tensors(scores, gt_matrix_valid2)
        metrics = cr.compute_all_recalls(preds, target, indexes)
        log.info(f"[{run_label}] Mixture/{partition}: {metrics}")

        descriptions = [
            str(r["query"]).strip().replace("for", "to")
            for _, r in row_meta_valid.iterrows()
        ]
        results = cr.build_ranked_results(
            scores, gt_matrix_valid2, gallery_valid, row_meta_valid, descriptions, "mixture"
        )

        out_path = out_dir / f"results_mixture_{partition}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved -> {out_path}")

        mixture_metrics[f"mixture_{partition}"] = {
            "caption_mode": "mixture",
            "category": "mixture",
            "partition": partition,
            "num_queries": len(row_meta_valid),
            "gallery_size": len(gallery_valid),
            **metrics,
        }

    summary_path = out_dir / "metrics_mixture_summary.json"
    with open(summary_path, "w") as f:
        json.dump(mixture_metrics, f, indent=2)
    log.info(f"Mixture metrics summary -> {summary_path}")

    print("\n" + "=" * 90)
    print(f"MIXTURE RESULTS — {run_label} — RECALL@K")
    print("=" * 90)
    print(f"{'Partition':<10}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'R@50':>7}")
    print("─" * 90)
    for key, mets in mixture_metrics.items():
        print(f"  {mets['partition']:<8}  "
              f"{mets.get('Recall@1',  0):>7.2f}  "
              f"{mets.get('Recall@5',  0):>7.2f}  "
              f"{mets.get('Recall@10', 0):>7.2f}  "
              f"{mets.get('Recall@50', 0):>7.2f}")
    print()

    del clip_model, tokenizer, preprocess
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return mixture_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    MIXTURE_OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    gt_data: dict[str, tuple[np.ndarray, list[str], pd.DataFrame]] = {}
    for partition in cr.PARTITIONS:
        gt_data[partition] = cr.load_ground_truth(partition)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    metrics_by_model: dict[str, dict[str, dict]] = {}
    for model_name, pretrained in cr.CLIP_MODELS:
        run_label = f"{model_name}_{pretrained}"
        metrics_by_model[run_label] = run_mixture_for_model(
            model_name, pretrained, gt_data, device
        )

    rows = []
    for run_label, all_metrics in metrics_by_model.items():
        for key, mets in all_metrics.items():
            rows.append({"model": run_label, **mets})

    combined_df = pd.DataFrame(rows)
    combined_csv_path = MIXTURE_OUT_DIR_BASE / "all_models_mixture_summary.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    log.info(f"\nCross-model mixture summary (csv)  -> {combined_csv_path}")

    combined_json_path = MIXTURE_OUT_DIR_BASE / "all_models_mixture_summary.json"
    with open(combined_json_path, "w") as f:
        json.dump(metrics_by_model, f, indent=2)
    log.info(f"Cross-model mixture summary (json) -> {combined_json_path}")

    print("\n" + "=" * 100)
    print("ALL MODELS — MIXTURE RECALL@K COMPARISON")
    print("=" * 100)
    print(f"{'Model':<28}  {'Partition':<10}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'R@50':>7}")
    print("─" * 100)
    for _, row in combined_df.iterrows():
        print(f"  {row['model']:<26}  {row['partition']:<10}  "
              f"{row.get('Recall@1',  0):>7.2f}  "
              f"{row.get('Recall@5',  0):>7.2f}  "
              f"{row.get('Recall@10', 0):>7.2f}  "
              f"{row.get('Recall@50', 0):>7.2f}")
    print()
    print(f"Per-model mixture result files under: {MIXTURE_OUT_DIR_BASE}/<model_name>_<pretrained>/")
    print(f"Cross-model mixture summary at      : {combined_csv_path}")
    print()


if __name__ == "__main__":
    main()