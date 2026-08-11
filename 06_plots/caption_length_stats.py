import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

OUTPUT_DIR = "/home/users/ymathansingh/Master_Thesis/06_plots/results/original"

# ----------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Simple whitespace/punctuation-aware word count."""
    tokens = re.findall(r"\b[\w'-]+\b", text)
    return len(tokens)


def load_all_caption_lengths():
    rows = []
    for label, fname in CAPTION_FILES.items():
        path = os.path.join(CAPTIONS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, caption in data.items():
            rows.append(
                {
                    "config": label,
                    "id": key,
                    "caption": caption,
                    "n_words": word_count(caption),
                    "n_chars": len(caption),
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_all_caption_lengths()

    # Summary statistics table (this is what you can drop straight into the thesis text/table)
    summary = (
        df.groupby("config")["n_words"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
        .reindex(list(CAPTION_FILES.keys()))
    )
    print("\n=== Caption length summary (in words) ===")
    print(summary)
    summary.to_csv(os.path.join(OUTPUT_DIR, "caption_length_stats.csv"))

    # Plot: box plot with individual points overlaid (more informative than box alone)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    order = list(CAPTION_FILES.keys())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(
        data=df,
        x="config",
        y="n_words",
        order=order,
        showfliers=True,
        width=0.55,
        palette="Set2",
        ax=ax,
    )

    # annotate mean on top of each box
    for i, cfg in enumerate(order):
        mean_val = summary.loc[cfg, "mean"]
        ax.text(
            i,
            df["n_words"].max() * 1.03,
            f"mean={mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("")
    ax.set_ylabel("Caption length (words)")
    ax.set_title("Distribution of generated caption length by pipeline configuration")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "caption_length_stats.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()