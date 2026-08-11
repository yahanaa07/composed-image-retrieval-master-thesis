"""
Catalan → English translation using Qwen2.5-7B-Instruct
"""

import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ── Config ──
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LOAD_IN_4BIT = False
BATCH_SIZE = 16

MAX_NEW_TOKENS = 128
DO_SAMPLE = True
TEMPERATURE = 0.1
TOP_P = 1.0
TOP_K = 50

# ── Load model ──
print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"

device = "cuda" if torch.cuda.is_available() else "cpu"

load_kwargs = {"trust_remote_code": True}

if device == "cuda":
    if LOAD_IN_4BIT:
        load_kwargs["load_in_4bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
else:
    load_kwargs["torch_dtype"] = torch.float32

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
if device == "cpu" or not LOAD_IN_4BIT:
    model = model.to(device) if device == "cpu" else model
model.eval()

# ── Word-order fix (IMPORTANT) ──
def normalize_phrase(text):
    words = text.split()
    if len(words) == 2:
        return f"{words[1]} {words[0]}"
    return text

# ── Prompt ──
SYSTEM_PROMPT = (
    "Translate Catalan museum taxonomy terms into English. "
    "Return only the English translation. "
    "Use correct English word order. "
    "Do not translate word-by-word."
)

def _make_messages(t):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": t}
    ]

# ── Batch translate ──
def _translate_batch(texts):
    if not texts:
        return []

    prompts = [
        tokenizer.apply_chat_template(
            _make_messages(t),
            tokenize=False,
            add_generation_prompt=True
        )
        for t in texts
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id
        )

    n = enc["input_ids"].shape[1]

    results = []
    for o in out:
        text = tokenizer.decode(o[n:], skip_special_tokens=True)
        text = text.split("\n")[0].strip()
        text = normalize_phrase(text)   # ✔ FIX APPLIED HERE
        results.append(text)

    return results

# ── Delimiters ──
_PIPE = re.compile(r'\s*\|\s*')
_DOLL = re.compile(r'\s*\$\s*')

# ── Build atom map ──
def build_atom_map(series_list):
    atoms = set()

    for s in series_list:
        for c in s.dropna():
            for d in _DOLL.split(str(c)):
                for p in _PIPE.split(d):
                    t = p.strip()
                    if t:
                        atoms.add(t)

    atoms = sorted(atoms)
    print(f"Atoms: {len(atoms)}")

    atom_map = {}

    for i in tqdm(range(0, len(atoms), BATCH_SIZE)):
        batch = atoms[i:i+BATCH_SIZE]
        translated = _translate_batch(batch)

        for k, v in zip(batch, translated):
            atom_map[k] = v

    return atom_map

# ── Translate cell ──
def translate_cell(cell, atom_map):
    if pd.isna(cell):
        return cell

    cell = str(cell).strip()
    if not cell:
        return cell

    out = []

    for d in _DOLL.split(cell):
        parts = _PIPE.split(d.strip())
        out.append(
            " | ".join(atom_map.get(p.strip(), p.strip()) for p in parts)
        )

    return " $ ".join(out)

# ── Main ──
def main():
    inp = "/home/users/ymathansingh/Thesis project/EUFCC-CIR/cir_db.csv"
    out = "/home/users/ymathansingh/Thesis project/02_translation_CIR/Results/cir_db_translated_qwen_1.0_50.csv"

    cols = [
        "materials_1", "ObjectTypes_1",
        "materials_2", "ObjectTypes_2",
        "element_to_change", "element_changed"
    ]

    df = pd.read_csv(inp)

    atom_map = build_atom_map([df[c] for c in cols if c in df.columns])

    for c in cols:
        if c in df.columns:
            print("col:", c)
            df[c] = df[c].apply(lambda x: translate_cell(x, atom_map))

    os.makedirs(os.path.dirname(out), exist_ok=True)

    df.to_csv(out, index=False)
    print("saved:", out)

if __name__ == "__main__":
    main()