"""
Shared utilities for caption generation scripts.
Imported by all four caption generation scripts:
  - generate_captions_first_hierarchy.py
  - generate_captions_full_hierarchy.py
  - generate_captions_mat_first_obj_full.py
  - generate_captions_mat_full_obj_first.py

Must be in the same directory as those scripts.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from PIL import Image as PILImage
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS   = 70    # ~50 words at 1.4 tokens/word
MAX_WORDS        = 50    # hard word cap (CLIP 77-token budget)
MIN_PIXELS       = 128 * 28 * 28
MAX_PIXELS       = 256 * 28 * 28
CACHE_SAVE_EVERY = 25

IMAGE_DIRS = [
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/test_id"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/test_ood"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/train"),
    Path("/home/users/ymathansingh/Master_Thesis/01_download_images/png/downloaded_images_png/val"),
]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Convert accented chars to ASCII. e.g. consommé → consomme."""
    n = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in n if not unicodedata.combining(c))


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalise_text(str(value).strip())


def enforce_word_limit(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    for punct in [".", "!", "?"]:
        idx = truncated.rfind(punct)
        if idx > len(truncated) // 2:
            return truncated[:idx + 1]
    return truncated + "."


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING — strip unwanted preamble and references to original
# ─────────────────────────────────────────────────────────────────────────────

_PREAMBLE_PATTERNS = [
    r'^the target image shows?\s*[:\-–]?\s*',
    r'^the target image depicts?\s*[:\-–]?\s*',
    r'^the image shows?\s*[:\-–]?\s*',
    r'^the image depicts?\s*[:\-–]?\s*',
    r'^this image shows?\s*[:\-–]?\s*',
    r'^the target is\s*[:\-–]?\s*',
    r'^the target shows?\s*[:\-–]?\s*',
    r'^the target depicts?\s*[:\-–]?\s*',
    r'^this shows?\s*[:\-–]?\s*',
    r'^shown is\s*[:\-–]?\s*',
    r'^depicted is\s*[:\-–]?\s*',
    r'^the picture shows?\s*[:\-–]?\s*',
    r'^the photo shows?\s*[:\-–]?\s*',
    r'^in the image[,\s]+',
    r'^in this image[,\s]+',
    r'^the result is\s*[:\-–]?\s*',
    r'^the resulting image shows?\s*[:\-–]?\s*',
]

_ORIGINAL_REF_PATTERNS = [
    r'\s*,?\s*similar (to|as) the (original|reference|dress|garment|previous|former)\w*',
    r'\s*,?\s*same (as|to) the (original|reference|dress|garment)\w*',
    r'\s*,?\s*maintaining the same \w+(\s+\w+)?\s*(as|to) the (original|dress)\w*',
    r'\s*,?\s*compared to the (original|reference|dress|garment)\w*',
    r'\s*,?\s*replacing the (original|reference)\w*',
    r'\s*,?\s*instead of the (original|previous)\w*',
    r'\s*,?\s*rather than the (original|previous)\w*',
]

def clean_description(text: str) -> str:
    """
    Remove preamble phrases and references to the original image.
    Capitalise the first letter of the cleaned result.

    Before: "The TARGET image shows a shirt with similar fabric to the dress."
    After:  "A shirt with similar fabric."
    """
    text = text.strip()

    for pat in _PREAMBLE_PATTERNS:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    for pat in _ORIGINAL_REF_PATTERNS:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*,\s*\.', '.', text)
    text = re.sub(r',\s*$', '.', text)

    if text:
        text = text[0].upper() + text[1:]

    return text


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PARSING
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_PATTERN = re.compile(r'^change\s+(.+?)\s+for\s+(.+?)\.?$', re.IGNORECASE)

def parse_query(query: str) -> tuple[str, str]:
    """
    Parse 'change X for Y' → (from_element, to_element).
    Returns ("", query) if pattern does not match.

    "change ceramic for brick."     →  ("ceramic", "brick")
    "change easel painting for lighting."  →  ("easel painting", "lighting")
    """
    query = normalise_text(query.strip())
    m = _QUERY_PATTERN.match(query)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", query


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHY PARSING
#
# PROFESSOR'S REQUIREMENT: "For empty cells in materials_1 and ObjectTypes_1
# — just don't add / do anything (to avoid extra noise)"
#
# Both parse functions return "" (empty string) when the field is empty/NaN.
# An empty string is then silently skipped inside build_messages() so that
# no "Reference image materials: " line appears in Qwen's prompt at all.
# Previously these returned "not specified", which added noise.
# ─────────────────────────────────────────────────────────────────────────────

def parse_first_hierarchy(raw) -> str:
    """
    First $ group, first | term → broadest label for primary object.
    Returns "" if the field is empty or NaN — caller must handle empty string.
    """
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    raw_str    = str(raw).strip()
    if not raw_str:
        return ""
    first_group = raw_str.split("$")[0].strip()
    first_term  = first_group.split("|")[0].strip()
    return normalise_text(first_term) if first_term else ""


def parse_full_hierarchy(raw) -> str:
    """
    All $ groups, last | term each → most specific label per object.
    Returns "" if the field is empty or NaN — caller must handle empty string.
    """
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    groups = [g.strip() for g in raw_str.split("$")]
    seen, terms = set(), []
    for group in groups:
        parts = [p.strip() for p in group.split("|")]
        term  = normalise_text(parts[-1])
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return ", ".join(terms) if terms else ""


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOOKUP WITH CACHING
# ─────────────────────────────────────────────────────────────────────────────

_path_cache: dict[str, Path | None] = {}
_img_cache:  dict[str, PILImage.Image] = {}


def find_image(image_id: str) -> Path | None:
    if image_id not in _path_cache:
        result = None
        for folder in IMAGE_DIRS:
            p = folder / f"{image_id}.png"
            if p.exists():
                result = p
                break
        _path_cache[image_id] = result
    return _path_cache[image_id]


def load_image(path: Path) -> PILImage.Image:
    key = str(path)
    if key not in _img_cache:
        _img_cache[key] = PILImage.open(path).convert("RGB")
    return _img_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You write short visual descriptions of objects for an image retrieval system. "
    "LANGUAGE: English only. "
    "FORMAT: Start directly with the object description. "
    "Do NOT start with 'The image shows', 'The TARGET image shows', "
    "'This shows', 'Depicted is', or any similar phrase. "
    f"LENGTH: Maximum {MAX_WORDS} words. One or two sentences. "
    "CONTENT: Describe only visual properties — material, texture, colour, "
    "surface finish, form, shape, structure. "
    "Use plain everyday English words. Avoid obscure technical or historical terms. "
    "STYLE: Write as if describing what you directly see in front of you. "
    "Example good output: "
    "'A rough red brick wall with rectangular clay blocks, deep mortar joints, "
    "and a matte earthy surface.' "
    "Example bad output: "
    "'The TARGET image shows a brick surface replacing the ceramic.' "
    "Never mention the original, what changed, or any transformation."
)

STRICT_PROMPT = (
    "Write one sentence describing an object. English only. "
    f"Maximum {MAX_WORDS} words. "
    "Start with 'A' or 'An' followed by the object description. "
    "Example: 'A smooth white porcelain bowl with a glossy glazed surface and rounded form.' "
    "Describe only: material, colour, texture, surface, form. "
    "No preamble. No mention of changes."
)

CHANGE_PATTERN = re.compile(
    r"\b(changed|replaced|transformed|originally|previously|"
    r"instead of|no longer|used to|now features|modified|altered|"
    r"has been|was replaced|was changed|transition|substitut)\b",
    re.IGNORECASE,
)


def build_messages(image_path, mat_str: str, obj_str: str,
                   target_element: str, system: str) -> list[dict]:
    """
    Build the Qwen-VL message list for one row.

    PROFESSOR'S REQUIREMENT: "For empty cells in materials_1 and ObjectTypes_1
    — just don't add / do anything (to avoid extra noise)"

    If mat_str or obj_str is "" (empty — returned by parse_*_hierarchy when
    the CSV field is NaN/empty), that context line is simply omitted from the
    prompt. Qwen never sees "Reference image materials: " with nothing after it.

    Examples:
      mat_str="ceramic", obj_str="jar"
        → "Reference image materials: ceramic\n
           Reference image object types: jar\n
           Describe this TARGET: brick\n..."

      mat_str="ceramic", obj_str=""          (ObjectTypes_1 was empty)
        → "Reference image materials: ceramic\n
           Describe this TARGET: brick\n..."

      mat_str="",        obj_str=""          (both empty)
        → "Describe this TARGET: brick\n..."
    """
    # Build context lines — only include non-empty fields
    context_lines: list[str] = []
    if mat_str:
        context_lines.append(f"Reference image materials: {mat_str}")
    if obj_str:
        context_lines.append(f"Reference image object types: {obj_str}")

    # Join with newline; add trailing newline only if there is any context
    context_block = "\n".join(context_lines)
    if context_block:
        context_block += "\n"

    user_text = (
        f"{context_block}"
        f"Describe this TARGET: {target_element}\n\n"
        f"Write a short visual description of {target_element} "
        f"in under {MAX_WORDS} words. "
        f"Start directly with the description — no preamble."
    )

    if image_path is not None:
        user_content = [
            {"type": "image", "image": load_image(image_path)},
            {"type": "text",  "text": user_text},
        ]
    else:
        user_content = user_text

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str, use_int8: bool):
    import logging
    log = logging.getLogger(__name__)

    log.info(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )

    if use_int8:
        log.info("Loading with INT8 quantization …")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        log.info("INT8 model loaded (~half VRAM of fp16)")
    else:
        log.info("Loading in fp16 …")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    log.info("Model ready.")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_one(image_path, mat_str: str, obj_str: str,
                 query: str, id1: str, model, processor, log) -> str:
    """
    Generate one target description.
    - Extracts target element from query (not the full transformation)
    - mat_str / obj_str are already parsed by the caller; empty string means
      the field was absent in the CSV — build_messages() will skip it silently
    - Retries with strict prompt if change-language or preamble detected
    - Post-processes to strip any remaining preamble
    """
    _, target_element = parse_query(query)

    def _run(system: str) -> str:
        messages = build_messages(image_path, mat_str, obj_str, target_element, system)
        text_in  = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if image_path is not None:
            inputs = processor(
                text=[text_in], images=[load_image(image_path)],
                return_tensors="pt", padding=True,
            ).to(model.device)
        else:
            inputs = processor(
                text=[text_in], return_tensors="pt", padding=True,
            ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        n = inputs["input_ids"].shape[1]
        return processor.tokenizer.decode(
            out[0][n:], skip_special_tokens=True).strip()

    raw = _run(SYSTEM_PROMPT)

    # Retry if change-language detected
    if CHANGE_PATTERN.search(raw):
        log.warning(f"  [{id1}] Change-language detected — retrying with strict prompt")
        raw = _run(STRICT_PROMPT)
        if CHANGE_PATTERN.search(raw):
            sentences = re.split(r'(?<=[.!?])\s+', raw)
            raw = " ".join(s for s in sentences if not CHANGE_PATTERN.search(s)) or raw

    # Retry if non-English detected
    if sum(1 for c in raw if ord(c) > 127) > 3:
        log.warning(f"  [{id1}] Non-English detected — retrying with strict prompt")
        raw = _run(STRICT_PROMPT)

    # Post-process: strip preamble + original references
    cleaned = clean_description(raw)

    # If cleaning made it too short (< 3 words), retry once more
    if len(cleaned.split()) < 3:
        cleaned = clean_description(_run(STRICT_PROMPT))

    return enforce_word_limit(cleaned, MAX_WORDS)