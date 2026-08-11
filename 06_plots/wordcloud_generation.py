import json
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_with_hierarchies/original/png/results_captions_original_png"
CAPTION_FILES = {
    "descriptions_first_hierarchy.json":   ("wordcloud_first_first",          "First / First"),
    "descriptions_full_hierarchy.json":    ("wordcloud_full_full",            "Full / Full"),
    "descriptions_mat_first_obj_full.json": ("wordcloud_mat_first_obj_full",  "Mat First / Obj Full"),
    "descriptions_mat_full_obj_first.json": ("wordcloud_mat_full_obj_first",  "Mat Full / Obj First"),
}

"""
CAPTIONS_DIR = "/home/users/ymathansingh/Master_Thesis/04_caption_generation_without_hierarchies/original/png/results_captions_original_png"
CAPTION_FILES = {
    "descriptions_no_hierarchy.json": ("wordcloud_no", "No"),
}
"""

OUTPUT_DIR = "/home/users/ymathansingh/Master_Thesis/06_plots/results/original"

# Effectively "no cap": with ~2,647 captions there are very likely more than 150
# unique non-stopword words, and the previous max_words=150 was silently dropping
# anything outside the top 150 (including words said only once). Raised high so
# every word that survives stopword-removal gets a slot in the cloud.
MAX_WORDS = 2000
# ----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s'-]", " ", text)  # drop punctuation/digits
    return text


def build_stopword_set():
    # Only the wordcloud library's built-in English stopword list,
    # plus the single word we explicitly want to exclude.
    stopwords = set(STOPWORDS)
    stopwords.add("featuring")
    return stopwords


def make_wordcloud(text_blob: str, title: str, out_path: str, stopwords: set):
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=stopwords,
        collocations=False,   # avoid duplicated bigrams inflating certain phrases
        max_words=MAX_WORDS,
        colormap="viridis",
    ).generate(text_blob)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stopwords = build_stopword_set()
    all_captions = []

    for fname, (out_stem, title) in CAPTION_FILES.items():
        path = os.path.join(CAPTIONS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        captions = [clean_text(c) for c in data.values()]
        all_captions.extend(captions)
        blob = " ".join(captions)

        out_path = os.path.join(OUTPUT_DIR, f"{out_stem}.png")
        make_wordcloud(blob, f"Word cloud — {title}", out_path, stopwords)


if __name__ == "__main__":
    print("File legend:")
    for fname, (out_stem, title) in CAPTION_FILES.items():
        print(f"  {out_stem}.png -> {title} ({fname})")
    print()
    main()