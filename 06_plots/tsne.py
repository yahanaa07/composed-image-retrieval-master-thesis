import os
import glob
import json
import random
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_with_hierarchies/original/png/results_captions_original_png"
CAPTION_FILE = "descriptions_full_hierarchy.json"  # pick the config you consider primary

"""
# No hierarchy
CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_without_hierarchies/original/png/results_captions_original_png"
CAPTION_FILE = "descriptions_no_hierarchy.json"  
"""

CSV_PATH = "/home/users/ymathansingh/Master_Thesis/03_CIR-dataset_updated/original/cir_db_updated_png.csv"
TARGET_ID_COLUMN = "id2"  # <-- VERIFY: column holding the target image id

IMAGES_ROOT = "/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png"
IMAGE_SUBFOLDERS = ["test_id", "test_ood", "train", "val"]
IMAGE_EXTENSIONS = ["*.png"]  

# CLIP backbone — match whatever you used as your best/main config in the thesis
CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"   # adjust the exact open_clip pretrained tag if needed

N_SAMPLES = 800          # cap sample size for t-SNE speed/readability; None = use all matched rows
BATCH_SIZE = 32
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "/home/users/ymathansingh/Master_Thesis/06_plots/results/original"
# ----------------------------------------------------------------------------


def build_image_index(images_root: str, subfolders, extensions):
    """Map filename stem (no extension) -> full path, searching across all subfolders."""
    index = {}
    for sub in subfolders:
        folder = os.path.join(images_root, sub)
        for ext in extensions:
            for path in glob.glob(os.path.join(folder, ext)):
                stem = os.path.splitext(os.path.basename(path))[0]
                index[stem] = path
    print(f"Indexed {len(index)} target images across {subfolders}")
    return index


def get_caption(captions: dict, row_idx: int):
    return captions.get(str(row_idx))


def encode_images(model, preprocess, paths, device, batch_size=32):
    embeddings = []
    valid_idx = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
            batch_paths = paths[start:start + batch_size]
            imgs = []
            local_valid = []
            for i, p in enumerate(batch_paths):
                try:
                    imgs.append(preprocess(Image.open(p).convert("RGB")))
                    local_valid.append(start + i)
                except Exception as e:
                    print(f"Skipping unreadable image {p}: {e}")
            if not imgs:
                continue
            batch = torch.stack(imgs).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            valid_idx.extend(local_valid)
    if embeddings:
        return np.concatenate(embeddings, axis=0), valid_idx
    return np.empty((0, 0)), []


def encode_texts(model, tokenizer, texts, device, batch_size=64):
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding captions"):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1) Load captions + CSV
    with open(os.path.join(CAPTIONS_DIR, CAPTION_FILE), "r", encoding="utf-8") as f:
        captions = json.load(f)

    df = pd.read_csv(CSV_PATH)

    # 2) Build target image index and resolve (caption, image_path) pairs
    image_index = build_image_index(IMAGES_ROOT, IMAGE_SUBFOLDERS, IMAGE_EXTENSIONS)

    pairs = []  # list of (caption_text, image_path)
    for row_idx, row in df.iterrows():
        caption = get_caption(captions, row_idx)
        if caption is None:
            continue
        target_id = str(row[TARGET_ID_COLUMN])
        image_path = image_index.get(target_id) or image_index.get(os.path.splitext(target_id)[0])
        if image_path is None:
            continue
        pairs.append((caption, image_path))

    print(f"Resolved {len(pairs)} (caption, target image) pairs out of {len(df)} CSV rows")

    if N_SAMPLES is not None and len(pairs) > N_SAMPLES:
        pairs = random.sample(pairs, N_SAMPLES)

    captions_list = [p[0] for p in pairs]
    image_paths = [p[1] for p in pairs]

    # 3) Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(DEVICE).eval()

    # 4) Encode
    image_embs, valid_idx = encode_images(model, preprocess, image_paths, DEVICE, BATCH_SIZE)
    captions_list = [captions_list[i] for i in valid_idx]  # keep only successfully-encoded pairs
    text_embs = encode_texts(model, tokenizer, captions_list, DEVICE, BATCH_SIZE)

    assert image_embs.shape[0] == text_embs.shape[0], "Mismatch between image and text embedding counts"

    # 5) t-SNE on combined embedding space
    combined = np.concatenate([image_embs, text_embs], axis=0)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init="pca", perplexity=30)
    coords_2d = tsne.fit_transform(combined)

    n = image_embs.shape[0]
    image_coords = coords_2d[:n]
    text_coords = coords_2d[n:]

    # 6) Modality gap: distance between the two cluster centroids
    image_centroid = image_coords.mean(axis=0)
    text_centroid = text_coords.mean(axis=0)
    modality_gap = np.linalg.norm(image_centroid - text_centroid)

    # 7) Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(image_coords[:, 0], image_coords[:, 1], s=10, alpha=0.6, color="#3B78C2", label="Target images")
    ax.scatter(text_coords[:, 0], text_coords[:, 1], s=10, alpha=0.6, color="#E0742B", label="Generated captions")

    for coords, color in [(image_coords, "#3B78C2"), (text_coords, "#E0742B")]:
        centroid = coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals) * 2  # ~2 std dev contour
        ellipse = Ellipse(centroid, width, height, angle=angle, edgecolor=color,
                           facecolor="none", linestyle="--", linewidth=1)
        ax.add_patch(ellipse)
        ax.scatter(*centroid, color="gold", edgecolor="black", s=60, zorder=5, marker="o")

    ax.set_title(f"t-SNE of target-image vs. generated-caption embeddings\nModality gap (MG) = {modality_gap:.2f}")
    ax.legend(loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "tsne.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()