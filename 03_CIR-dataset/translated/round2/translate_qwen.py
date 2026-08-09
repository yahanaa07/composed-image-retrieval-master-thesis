"""
Catalan -> English translation using Qwen2.5-14B-Instruct
Manual corrections are loaded from manual_corrections.csv (edit that file,
not this script, when you find new wrong translations).
"""

import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ── Config ──
MODEL_ID      = "Qwen/Qwen2.5-14B-Instruct"
LOAD_IN_4BIT  = True
BATCH_SIZE    = 8
MAX_NEW_TOKENS = 32    # taxonomy labels are rarely longer than 5 tokens;
                       # 32 is plenty and saves significant compute vs 128
DO_SAMPLE     = False  # greedy: deterministic and more accurate for translation
TEMPERATURE   = 1.0    # ignored when DO_SAMPLE=False, kept for completeness
TOP_P         = 1.0    # ignored when DO_SAMPLE=False
TOP_K         = 1      # ignored when DO_SAMPLE=False

# Paths
INPUT_CSV = (
    "/home/users/ymathansingh/Master_Thesis/"
    "03_CIR-dataset_updated/original/"
    "cir_db_updated_jpg.csv"
)

OUTPUT_CSV = (
    "/home/users/ymathansingh/Master_Thesis/"
    "03_CIR-dataset_updated/translated/jpg/"
    "cir_db_updated_jpg_translated.csv"
)

CORRECTIONS_CSV = (
    "/home/users/ymathansingh/Master_Thesis/"
    "03_CIR-dataset/translated/round2/"
    "manual_corrections_v5.csv"
)

COLS = [
    "materials_1", "ObjectTypes_1",
    "materials_2", "ObjectTypes_2",
    "element_to_change", "element_changed",
]

# ── Load manual corrections ──
def load_corrections(path: str) -> dict:
    """
    Reads manual_corrections.csv (two columns: catalan, english).
    Returns a plain dict {catalan_term: english_term}.
    Edit manual_corrections.csv whenever you spot a wrong model translation —
    no need to touch this script.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    corrections = dict(zip(df["catalan"].str.strip(), df["english"].str.strip()))
    print(f"Loaded {len(corrections)} manual corrections from {path}")
    return corrections

# ── Load model ──
def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.padding_side = "left"

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        if LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    mdl.eval()
    return tok, mdl

# ── Prompt ──
SYSTEM_PROMPT = """You are a specialist translator for Catalan museum and art history catalogue terms.
Translate the given Catalan term into English.
Return ONLY the English term — no explanation, no alternatives, no punctuation.
Use correct English word order (adjective before noun).
These are taxonomy labels for materials, object types, and art techniques.

Examples:
engalba → slip
sostre → ceiling
maó → brick
or → gold
caolí → kaolin
llim → silt
nu → nude
gravat → engraving
atuell → vessel
brusa → blouse
gresol → crucible
teler → loom
relleu → relief
cotilla → corset
"""

def _make_messages(term: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": term},
    ]

# ── Batch translate (model only) ──
def _translate_batch(texts: list, tokenizer, model) -> list:
    device = next(model.parameters()).device

    prompts = [
        tokenizer.apply_chat_template(
            _make_messages(t), tokenize=False, add_generation_prompt=True
        )
        for t in texts
    ]

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    results = []
    for o in out:
        text = tokenizer.decode(o[input_len:], skip_special_tokens=True)
        text = text.split("\n")[0].strip().strip('"\'')
        results.append(text)
    return results

# ── Delimiters ──
_PIPE = re.compile(r'\s*\|\s*')
_DOLL = re.compile(r'\s*\$\s*')

# ── Build atom map ──
def build_atom_map(series_list: list, corrections: dict, tokenizer, model) -> dict:
    """
    1. Collect every unique leaf term (atom) from all target columns.
    2. Apply manual corrections directly — no model call needed for those.
    3. Send the rest to the model in batches.
    4. Return a flat dict: {catalan_atom -> english_atom}.
    """
    atoms: set = set()
    for s in series_list:
        for cell in s.dropna():
            for segment in _DOLL.split(str(cell)):
                for part in _PIPE.split(segment):
                    t = part.strip()
                    if t:
                        atoms.add(t)

    manual_hits   = {a: corrections[a] for a in atoms if a in corrections}
    model_targets = sorted(a for a in atoms if a not in corrections)

    print(f"Unique atoms        : {len(atoms)}")
    print(f"  manual corrections: {len(manual_hits)}")
    print(f"  sent to model     : {len(model_targets)}")

    atom_map = dict(manual_hits)  # start with the guaranteed-correct ones

    for i in tqdm(range(0, len(model_targets), BATCH_SIZE), desc="Model translating"):
        batch = model_targets[i : i + BATCH_SIZE]
        translated = _translate_batch(batch, tokenizer, model)
        atom_map.update(zip(batch, translated))

    return atom_map

# ── Translate one CSV cell ──
def translate_cell(cell, atom_map: dict) -> str:
    if pd.isna(cell):
        return cell
    cell = str(cell).strip()
    if not cell:
        return cell

    segments = []
    for segment in _DOLL.split(cell):
        parts = _PIPE.split(segment.strip())
        segments.append(
            " | ".join(atom_map.get(p.strip(), p.strip()) for p in parts)
        )
    return " $ ".join(segments)

# ── Main ──
def main():
    corrections        = load_corrections(CORRECTIONS_CSV)
    tokenizer, model   = load_model()

    df = pd.read_csv(INPUT_CSV)
    atom_map = build_atom_map(
        [df[c] for c in COLS if c in df.columns],
        corrections, tokenizer, model,
    )

    # Create the output folder (and any missing parents) before writing anything.
    # exist_ok=True means no error if the folder already exists.
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # ── Save atom map for review ──
    # This CSV lists every unique Catalan leaf term and what it was translated to.
    # Open it after a run, scan for mistakes, add them to manual_corrections.csv,
    # and they will be fixed automatically on the next run without re-running the model.
    atom_map_path = OUTPUT_CSV.replace(".csv", "_atom_map.csv")
    (
        pd.DataFrame(sorted(atom_map.items()), columns=["catalan", "english"])
        .to_csv(atom_map_path, index=False)
    )
    print(f"Atom map saved     -> {atom_map_path}")
    print("  Open it, scan for wrong translations, add them to manual_corrections.csv")

    # ── Apply translations to all target columns ──
    for col in COLS:
        if col in df.columns:
            print(f"Translating column : {col}")
            df[col] = df[col].apply(lambda x: translate_cell(x, atom_map))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Results saved -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()