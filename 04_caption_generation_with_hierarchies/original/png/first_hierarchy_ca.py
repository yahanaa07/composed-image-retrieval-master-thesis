import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from shared_utils_ca import (
    clean, enforce_word_limit, find_image, generate_one,
    load_model, parse_first_hierarchy_ca, parse_query,
    CACHE_SAVE_EVERY, MAX_WORDS,
)

# PATHS
CSV_PATH = Path(
    "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset_updated/original/"
    "cir_db_updated_png.csv"
)

CAPTIONS_OUT_DIR = Path(
    "/home/users/ymathansingh/Master_Thesis/04_caption_generation_with_hierarchies/original/png/"
    "results_captions_original_png"
)

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
USE_INT8 = True

# Language of the generated captions for THIS pipeline (Catalan).
# The `query` column itself stays English (see shared_utils_ca.py docstring) —
# only the hierarchy context and the generated description are Catalan.
LANG = "ca"

CACHE_FILE = CAPTIONS_OUT_DIR / "descriptions_first_hierarchy.json"
ENRICHED_CSV = CAPTIONS_OUT_DIR / "enriched_first_hierarchy.csv"


def ensure_directories():
    CAPTIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENRICHED_CSV.parent.mkdir(parents=True, exist_ok=True)


ensure_directories()

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():

    log.info(f"Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    log.info(f"Rows: {len(df)} | partitions: {df['partition'].value_counts().to_dict()}")

    # Image availability
    found = sum(1 for iid in df["id1"].unique() if find_image(str(iid)))
    log.info(f"Reference images: {found} / {df['id1'].nunique()} found")

    # Show what the parsed context looks like
    sample = df.iloc[0]
    _, tgt = parse_query(clean(sample.get("query", "")))
    log.info("Example — what Qwen will receive (CATALAN captions, mat/obj=FIRST):")
    log.info(f"  query         : {sample['query']}")
    log.info(f"  target element: {tgt}  ← this replaces the full query (still English — see shared_utils_ca.py)")
    log.info(f"  mat (first,ca): {parse_first_hierarchy_ca(sample.get('materials_1',''))}")
    log.info(f"  obj (first,ca): {parse_first_hierarchy_ca(sample.get('ObjectTypes_1',''))}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info(f"  GPU: {p.name}  {p.total_memory/1e9:.1f} GB")

    # Load cache
    cache: dict[str, str] = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        log.info(f"Resuming: {len(cache)} already cached")

    # Load model
    model, processor = load_model(QWEN_MODEL_ID, USE_INT8)

    # Generate
    todo = [i for i in df.index if str(i) not in cache]
    log.info(f"Rows to generate: {len(todo)}")
    log.info(f"Estimated time  : {len(todo) * 5 / 3600:.1f} h  (at ~5s/row)")

    for step, idx in enumerate(tqdm(todo, desc="[first hierarchy - ca]")):
        row     = df.iloc[idx]
        id1     = clean(row.get("id1", ""))
        query   = clean(row.get("query", ""))
        mat_str = parse_first_hierarchy_ca(row.get("materials_1", ""))
        obj_str = parse_first_hierarchy_ca(row.get("ObjectTypes_1", ""))
        img_p   = find_image(id1)

        try:
            desc = generate_one(img_p, mat_str, obj_str, query, id1,
                                 model, processor, log, lang=LANG)
        except Exception as e:
            log.error(f"Row {idx} ({id1}): {e}")
            desc = ""

        cache[str(idx)] = desc

        if (step + 1) % CACHE_SAVE_EVERY == 0:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(cache)} descriptions → {CACHE_FILE}")

    df["desc_first_hierarchy"] = [cache.get(str(i), "") for i in df.index]
    df.to_csv(ENRICHED_CSV, index=False)
    log.info(f"Enriched CSV → {ENRICHED_CSV}")

    wc   = df["desc_first_hierarchy"].dropna().apply(lambda x: len(str(x).split()))
    over = (wc > MAX_WORDS).sum()
    log.info(f"Word counts — mean:{wc.mean():.1f}  max:{wc.max()}  over_limit:{over}")

    # Verification
    print("\n" + "="*70)
    print("FIRST HIERARCHY (CATALAN) — VERIFICATION SAMPLE")
    print(f"Word limit: {MAX_WORDS}  |  Format: pure visual description of target, in Catalan")
    print("="*70)
    for part in sorted(df["partition"].unique()):
        print(f"\n--- {part} ---")
        for _, row in df[df["partition"] == part].head(3).iterrows():
            img = find_image(str(row["id1"]))
            _, tgt = parse_query(clean(row.get("query", "")))
            d = str(row["desc_first_hierarchy"])
            print(f"  id1       : {row['id1']}  ({'image' if img else 'TEXT FALLBACK'})")
            print(f"  query     : {row['query']}")
            print(f"  target    : {tgt}  ← Qwen described THIS (in Catalan)")
            print(f"  mat ctx   : {parse_first_hierarchy_ca(str(row.get('materials_1','')))}")
            print(f"  desc      : ({len(d.split())}w) {d}")
            print()

if __name__ == "__main__":
    main()