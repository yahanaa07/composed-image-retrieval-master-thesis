import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
from tqdm import tqdm

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_with_hierarchies/original/png/results_captions_original_png"
CAPTION_FILES = {
    "first_hierarchy": "descriptions_first_hierarchy.json",
    "full_hierarchy": "descriptions_full_hierarchy.json",
    "mat_first_obj_full": "descriptions_mat_first_obj_full.json",
    "mat_full_obj_first": "descriptions_mat_full_obj_first.json",
}

"""
CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_without_hierarchies/original/png/results_captions_original_png"
CAPTION_FILES = {
    "no_hierarchy": "descriptions_no_hierarchy.json",
}
"""

CSV_PATH = "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset_updated/original/cir_db_updated_png.csv"
REFERENCE_ID_COLUMN = "id1"
TARGET_ID_COLUMN = "id2"
QUERY_TEXT_COLUMN = "query"

IMAGES_ROOT = "/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png"
IMAGE_SUBFOLDERS = ["test_id", "test_ood", "train", "val"]
IMAGE_EXTENSIONS = ["*.png"]

CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"

N_SAMPLES = 1000
BATCH_SIZE = 32
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "/home/users/ymathansingh/Master_Thesis/06_plots/results/original"
# ----------------------------------------------------------------------------


def build_image_index(images_root, subfolders, extensions):
    index = {}
    for sub in subfolders:
        folder = os.path.join(images_root, sub)
        for ext in extensions:
            for path in glob.glob(os.path.join(folder, ext)):
                stem = os.path.splitext(os.path.basename(path))[0]
                index[stem] = path
    print(f"Indexed {len(index)} images across {subfolders}")
    return index


def get_caption(captions, row_idx):
    return captions.get(str(row_idx))


def encode_images(model, preprocess, paths, device, batch_size=32):
    embeddings = np.full((len(paths), model.visual.output_dim), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
            batch_idx = list(range(start, min(start + batch_size, len(paths))))
            imgs, valid_local = [], []
            for i in batch_idx:
                try:
                    imgs.append(preprocess(Image.open(paths[i]).convert("RGB")))
                    valid_local.append(i)
                except Exception as e:
                    print(f"Skipping unreadable image {paths[i]}: {e}")
            if not imgs:
                continue
            batch = torch.stack(imgs).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()
            for j, idx in enumerate(valid_local):
                embeddings[idx] = feats[j]
    return embeddings


def encode_texts(model, tokenizer, texts, device, batch_size=64):
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cosine_sim_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def combine_captions_embed(per_config_embs: dict) -> np.ndarray:
    stacked = np.stack(list(per_config_embs.values()), axis=0)
    combined = stacked.mean(axis=0)
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    return combined


def plot_raincloud(ax, data_list, labels, colors,
                    cloud_height=0.38, box_width=0.16, rain_offset=0.28, jitter=0.11,
                    rng=None):
    """Draw a horizontal raincloud: half-violin (cloud) above each category's
    baseline, a thin box plot at the baseline, and jittered points (rain) below."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(data_list)
    for i, (vals, color) in enumerate(zip(data_list, colors)):
        y0 = n - i - 1  # first category ends up on top of the plot

        # --- Cloud: half-violin (KDE), bulging upward from the baseline ---
        vals = np.asarray(vals)
        if len(vals) > 1 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            pad = (vals.max() - vals.min()) * 0.15 or 0.05
            xs = np.linspace(vals.min() - pad, vals.max() + pad, 300)
            dens = kde(xs)
            dens = dens / dens.max() * cloud_height
        else:
            xs = np.array([vals.min(), vals.max()])
            dens = np.zeros_like(xs)
        ax.fill_between(xs, y0, y0 + dens, color=color, alpha=0.7,
                         edgecolor="black", linewidth=0.6, zorder=2)

        # --- Box: thin horizontal box plot sitting on the baseline ---
        ax.boxplot(
            vals, positions=[y0], vert=False, widths=box_width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.9, edgecolor="black", linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.3),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            zorder=3,
        )

        # --- Rain: jittered individual points below the baseline ---
        y_jitter = y0 - rain_offset + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(vals, y_jitter, s=4, color=color, alpha=0.25, zorder=1, edgecolor="none")

    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_ylim(-1.0, n - 1 + cloud_height + 0.35)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    captions_by_config = {}
    for label, fname in CAPTION_FILES.items():
        with open(os.path.join(CAPTIONS_DIR, fname), "r", encoding="utf-8") as f:
            captions_by_config[label] = json.load(f)

    df = pd.read_csv(CSV_PATH)
    image_index = build_image_index(IMAGES_ROOT, IMAGE_SUBFOLDERS, IMAGE_EXTENSIONS)

    records = []
    for row_idx, row in df.iterrows():
        caps = {}
        skip = False
        for label, caption_dict in captions_by_config.items():
            c = get_caption(caption_dict, row_idx)
            if c is None:
                skip = True
                break
            caps[label] = c
        if skip:
            continue

        ref_id = str(row[REFERENCE_ID_COLUMN])
        tgt_id = str(row[TARGET_ID_COLUMN])
        query_text = row[QUERY_TEXT_COLUMN]

        ref_path = image_index.get(ref_id) or image_index.get(os.path.splitext(ref_id)[0])
        tgt_path = image_index.get(tgt_id) or image_index.get(os.path.splitext(tgt_id)[0])

        if ref_path is None or tgt_path is None or not isinstance(query_text, str):
            continue

        records.append(
            {"query": query_text, "ref_path": ref_path, "tgt_path": tgt_path, "captions": caps}
        )

    print(f"Resolved {len(records)} complete rows out of {len(df)} CSV rows")

    if N_SAMPLES is not None and len(records) > N_SAMPLES:
        idx = np.random.choice(len(records), N_SAMPLES, replace=False)
        records = [records[i] for i in idx]

    queries = [r["query"] for r in records]
    ref_paths = [r["ref_path"] for r in records]
    tgt_paths = [r["tgt_path"] for r in records]

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(DEVICE).eval()

    query_embs = encode_texts(model, tokenizer, queries, DEVICE, BATCH_SIZE)
    ref_embs = encode_images(model, preprocess, ref_paths, DEVICE, BATCH_SIZE)
    tgt_embs = encode_images(model, preprocess, tgt_paths, DEVICE, BATCH_SIZE)

    caption_embs_by_config = {}
    for label in CAPTION_FILES.keys():
        texts = [r["captions"][label] for r in records]
        caption_embs_by_config[label] = encode_texts(model, tokenizer, texts, DEVICE, BATCH_SIZE)

    combined_embs = combine_captions_embed(caption_embs_by_config)

    valid_mask = ~(np.isnan(ref_embs).any(axis=1) | np.isnan(tgt_embs).any(axis=1))
    query_embs, ref_embs, tgt_embs = query_embs[valid_mask], ref_embs[valid_mask], tgt_embs[valid_mask]
    combined_embs = combined_embs[valid_mask]
    for label in caption_embs_by_config:
        caption_embs_by_config[label] = caption_embs_by_config[label][valid_mask]
    print(f"Final valid sample size: {valid_mask.sum()} / {len(valid_mask)}")

    sims = {
        "Query text": cosine_sim_rowwise(query_embs, tgt_embs),
        "Reference image (id1)": cosine_sim_rowwise(ref_embs, tgt_embs),
    }
    for label, embs in caption_embs_by_config.items():
        sims[label] = cosine_sim_rowwise(embs, tgt_embs)
    order = [
        "Query text",
        "Reference image (id1)",
        "first_hierarchy",
        "full_hierarchy",
        "mat_first_obj_full",
        "mat_full_obj_first",
    ]


    means = {label: np.mean(sims[label]) for label in order}
    print("\n=== Average cosine similarity per input type (vs. target image) ===")
    for label in order:
        print(f"{label:35s} mean={means[label]:.4f}  n={len(sims[label])}")

    # --- Plot: raincloud ---
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#B3B3B3", "#B3B3B3", "#8DA0CB", "#8DA0CB", "#8DA0CB", "#8DA0CB", "#66C2A5"]

    data_list = [sims[label] for label in order]

    plot_raincloud(ax, data_list, order, colors)

    # Mean markers + labels
    n = len(order)
    for i, label in enumerate(order):
        y0 = n - i - 1
        ax.scatter([means[label]], [y0], marker="D", color="black", s=22, zorder=4)
        ax.text(means[label], y0 + 0.42, f"mean={means[label]:.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Cosine similarity with target image")
    ax.set_ylabel("")
    ax.set_title("Cosine similarity with target image: query, reference image, and generated captions\n(raincloud: density + box plot + individual points)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cosine_similarity_raincloud.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()


# No hierarchy
"""
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    captions_by_config = {}
    for label, fname in CAPTION_FILES.items():
        with open(os.path.join(CAPTIONS_DIR, fname), "r", encoding="utf-8") as f:
            captions_by_config[label] = json.load(f)

    df = pd.read_csv(CSV_PATH)
    image_index = build_image_index(
        IMAGES_ROOT,
        IMAGE_SUBFOLDERS,
        IMAGE_EXTENSIONS
    )

    records = []

    for row_idx, row in df.iterrows():
        caps = {}
        skip = False

        for label, caption_dict in captions_by_config.items():
            c = get_caption(caption_dict, row_idx)

            if c is None:
                skip = True
                break

            caps[label] = c

        if skip:
            continue

        ref_id = str(row[REFERENCE_ID_COLUMN])
        tgt_id = str(row[TARGET_ID_COLUMN])
        query_text = row[QUERY_TEXT_COLUMN]

        ref_path = (
            image_index.get(ref_id)
            or image_index.get(os.path.splitext(ref_id)[0])
        )

        tgt_path = (
            image_index.get(tgt_id)
            or image_index.get(os.path.splitext(tgt_id)[0])
        )

        if (
            ref_path is None
            or tgt_path is None
            or not isinstance(query_text, str)
        ):
            continue

        records.append(
            {
                "query": query_text,
                "ref_path": ref_path,
                "tgt_path": tgt_path,
                "captions": caps,
            }
        )

    print(
        f"Resolved {len(records)} complete rows "
        f"out of {len(df)} CSV rows"
    )

    if N_SAMPLES is not None and len(records) > N_SAMPLES:
        idx = np.random.choice(
            len(records),
            N_SAMPLES,
            replace=False
        )
        records = [records[i] for i in idx]

    queries = [r["query"] for r in records]
    ref_paths = [r["ref_path"] for r in records]
    tgt_paths = [r["tgt_path"] for r in records]

    # ------------------------------------------------------------------------
    # CLIP model
    # ------------------------------------------------------------------------

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    model = model.to(DEVICE).eval()

    # ------------------------------------------------------------------------
    # Encode query, reference and target images
    # ------------------------------------------------------------------------

    query_embs = encode_texts(
        model,
        tokenizer,
        queries,
        DEVICE,
        BATCH_SIZE
    )

    ref_embs = encode_images(
        model,
        preprocess,
        ref_paths,
        DEVICE,
        BATCH_SIZE
    )

    tgt_embs = encode_images(
        model,
        preprocess,
        tgt_paths,
        DEVICE,
        BATCH_SIZE
    )

    # ------------------------------------------------------------------------
    # Encode captions
    # ------------------------------------------------------------------------

    caption_embs_by_config = {}

    for label in CAPTION_FILES.keys():
        texts = [r["captions"][label] for r in records]

        caption_embs_by_config[label] = encode_texts(
            model,
            tokenizer,
            texts,
            DEVICE,
            BATCH_SIZE
        )

    combined_embs = combine_captions_embed(
        caption_embs_by_config
    )

    # ------------------------------------------------------------------------
    # Remove samples with invalid image embeddings
    # ------------------------------------------------------------------------

    valid_mask = ~(
        np.isnan(ref_embs).any(axis=1)
        | np.isnan(tgt_embs).any(axis=1)
    )

    query_embs = query_embs[valid_mask]
    ref_embs = ref_embs[valid_mask]
    tgt_embs = tgt_embs[valid_mask]
    combined_embs = combined_embs[valid_mask]

    for label in caption_embs_by_config:
        caption_embs_by_config[label] = (
            caption_embs_by_config[label][valid_mask]
        )

    print(
        f"Final valid sample size: "
        f"{valid_mask.sum()} / {len(valid_mask)}"
    )

    # ------------------------------------------------------------------------
    # Cosine similarities
    # ------------------------------------------------------------------------

    sims = {
        "Query text": cosine_sim_rowwise(
            query_embs,
            tgt_embs
        ),
        "Reference image (id1)": cosine_sim_rowwise(
            ref_embs,
            tgt_embs
        ),
    }

    for label, embs in caption_embs_by_config.items():
        sims[label] = cosine_sim_rowwise(
            embs,
            tgt_embs
        )

    # ------------------------------------------------------------------------
    # Order of categories
    # ------------------------------------------------------------------------

    order = [
        "Query text",
        "Reference image (id1)",
        "no_hierarchy",
    ]

    # ------------------------------------------------------------------------
    # Mean cosine similarity
    # ------------------------------------------------------------------------

    means = {
        label: np.mean(sims[label])
        for label in order
    }

    print(
        "\n=== Average cosine similarity per input type "
        "(vs. target image) ==="
    )

    for label in order:
        print(
            f"{label:35s} "
            f"mean={means[label]:.4f}  "
            f"n={len(sims[label])}"
        )

    # ------------------------------------------------------------------------
    # Plot: raincloud
    # ------------------------------------------------------------------------

    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        "#B3B3B3",
        "#B3B3B3",
        "#66C2A5",
    ]

    data_list = [
        sims[label]
        for label in order
    ]

    plot_raincloud(
        ax,
        data_list,
        order,
        colors
    )

    # ------------------------------------------------------------------------
    # Mean markers + labels
    # ------------------------------------------------------------------------

    n = len(order)

    for i, label in enumerate(order):
        y0 = n - i - 1

        ax.scatter(
            [means[label]],
            [y0],
            marker="D",
            color="black",
            s=22,
            zorder=4
        )

        ax.text(
            means[label],
            y0 + 0.42,
            f"mean={means[label]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )

    # ------------------------------------------------------------------------
    # Plot formatting
    # ------------------------------------------------------------------------

    ax.set_xlabel(
        "Cosine similarity with target image"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Cosine similarity with target image: "
        "query, reference image, and generated captions\n"
        "(raincloud: density + box plot + individual points)"
    )

    ax.xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=8)
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        "cosine_similarity_raincloud.png"
    )

    plt.savefig(
        out_path,
        dpi=300
    )

    print(
        f"\nSaved figure to {out_path}"
    )


if __name__ == "__main__":
    main()
"""