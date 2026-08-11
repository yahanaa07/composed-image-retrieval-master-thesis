import csv
import json
from pathlib import Path

# Must match the Catalan clip_retrieval.py's CLIP_OUT_DIR_BASE exactly.
CLIP_OUT_DIR_BASE = Path(
    "/home/users/ymathansingh/Master_Thesis/05_clip_retrieval/original/png/"
    "results_clip_retrieval_ca"
)

# The 10 test_ood row_idx values whose query sentence collapsed to
# "change X for X" (same word on both sides), so their caption was
# generated from a generic label instead of a more specific one. Unchanged
# from the translated run — see the module docstring for why this should
# still be the correct set of row_idx values.
AFFECTED_ROW_IDX = {374, 375, 376, 377, 378, 379, 380, 381, 382, 383}

RECALL_KS = [1, 5, 10, 50]


def check_file(results_path: Path) -> list[dict]:
    with open(results_path) as f:
        results = json.load(f)
    return [r for r in results if r["row_idx"] in AFFECTED_ROW_IDX]


def main():
    if not CLIP_OUT_DIR_BASE.is_dir():
        print(f"CLIP_OUT_DIR_BASE not found: {CLIP_OUT_DIR_BASE}\n"
              f"Check the path matches clip_retrieval.py's CLIP_OUT_DIR_BASE.")
        return

    model_dirs = sorted(d for d in CLIP_OUT_DIR_BASE.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No model subfolders found under {CLIP_OUT_DIR_BASE} "
              f"— has clip_retrieval.py produced any results yet?")
        return

    header = (f"{'Model':<28} {'Mode':<14} {'row_idx':>7} {'id1':<18} "
              + "".join(f"{'hit@'+str(k):>8}" for k in RECALL_KS)
              + f" {'first_rank':>11}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for model_dir in model_dirs:
        result_files = sorted(model_dir.glob("results_*_test_ood.json"))
        for rp in result_files:
            # filename pattern: results_{mode}_test_ood.json
            mode = rp.stem.replace("results_", "").replace("_test_ood", "")
            for r in check_file(rp):
                gt = set(r["ground_truth"])
                ranked = r["ranked_ids"]
                hits = {k: bool(gt & set(ranked[:k])) for k in RECALL_KS}
                first_hit = next((i + 1 for i, x in enumerate(ranked) if x in gt), None)
                print(f"{model_dir.name:<28} {mode:<14} {r['row_idx']:>7} "
                      f"{r['id1']:<18} "
                      + "".join(f"{str(hits[k]):>8}" for k in RECALL_KS)
                      + f" {str(first_hit):>11}")
                summary_rows.append({
                    "model": model_dir.name,
                    "mode": mode,
                    "row_idx": r["row_idx"],
                    "id1": r["id1"],
                    **{f"hit@{k}": hits[k] for k in RECALL_KS},
                    "first_correct_rank": first_hit,
                })

    if not summary_rows:
        print("\nNo matching rows found in any results_*_test_ood.json file. "
              "Either clip_retrieval.py hasn't produced test_ood results yet, "
              "or AFFECTED_ROW_IDX doesn't match this dataset — see the module "
              "docstring's note about verifying id1 at row_idx 374 matches "
              "between the two CSVs.")
        return

    out_csv = CLIP_OUT_DIR_BASE / "affected_queries_check.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved -> {out_csv}")

    n = len(summary_rows)
    for k in RECALL_KS:
        hit_n = sum(1 for r in summary_rows if r[f"hit@{k}"])
        print(f"  hit@{k:<2}: {hit_n}/{n} affected-query result rows "
              f"({hit_n/n*100:.1f}%)")


if __name__ == "__main__":
    main()