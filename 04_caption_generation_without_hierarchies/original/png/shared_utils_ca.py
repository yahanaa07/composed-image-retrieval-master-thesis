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
MAX_NEW_TOKENS    = 70    # English:  ~50 words at ~1.4 tokens/word
MAX_NEW_TOKENS_CA = 90    # Catalan:  same 50-word cap, but Qwen's tokenizer
                          # splits Catalan into more subword tokens per word
                          # than English, so give it more headroom before
                          # enforce_word_limit() truncates back to 50 words.
MAX_WORDS         = 50    # hard word cap (CLIP 77-token budget) — same for both
MIN_PIXELS        = 128 * 28 * 28
MAX_PIXELS        = 256 * 28 * 28
CACHE_SAVE_EVERY  = 25
 
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
    """Convert accented chars to ASCII. e.g. consommé → consomme.
 
    NOTE: this strips ALL diacritics (NFKD + drop combining marks). Correct
    for the English/translated pipeline. Do NOT use this on genuine Catalan
    text — use the *_ca parsing helpers below instead, which preserve
    accents.
    """
    n = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in n if not unicodedata.combining(c))
 
 
def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalise_text(str(value).strip())
 
 
def clean_keep_accents(value) -> str:
    """Same as clean(), but does NOT strip accents. Used for Catalan text
    fields (materials_1 / ObjectTypes_1 raw values) where diacritics are
    part of correct spelling. id1 and query are ASCII/English already, so
    the regular clean() remains fine for those in the Catalan scripts too.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
 
 
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
# POST-PROCESSING (ENGLISH) — strip unwanted preamble and references to original
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
# POST-PROCESSING (CATALAN) — mirrors clean_description() above
# ─────────────────────────────────────────────────────────────────────────────
 
_PREAMBLE_PATTERNS_CA = [
    r'^la imatge objectiu mostra\s*[:\-–]?\s*',
    r'^la imatge objectiu representa\s*[:\-–]?\s*',
    r'^la imatge mostra\s*[:\-–]?\s*',
    r'^la imatge representa\s*[:\-–]?\s*',
    r'^aquesta imatge mostra\s*[:\-–]?\s*',
    r'^aquesta imatge representa\s*[:\-–]?\s*',
    r'^l[\'’]objectiu (es|és)\s*[:\-–]?\s*',
    r'^l[\'’]objectiu mostra\s*[:\-–]?\s*',
    r'^es mostra\s*[:\-–]?\s*',
    r'^es representa\s*[:\-–]?\s*',
    r'^la fotografia mostra\s*[:\-–]?\s*',
    r'^la imatge resultant mostra\s*[:\-–]?\s*',
    r'^el resultat (es|és)\s*[:\-–]?\s*',
    r'^en la imatge[,\s]+',
    r'^en aquesta imatge[,\s]+',
]
 
_ORIGINAL_REF_PATTERNS_CA = [
    r'\s*,?\s*semblant a l[\'’](original|imatge original|referencia|referència)\w*',
    r'\s*,?\s*igual (que|a) l[\'’](original|referencia|referència)\w*',
    r'\s*,?\s*en comparacio amb l[\'’](original|referencia)\w*',
    r'\s*,?\s*en comparació amb l[\'’](original|referència)\w*',
    r'\s*,?\s*en lloc de l[\'’](original|anterior)\w*',
    r'\s*,?\s*substituint l[\'’](original|referencia|referència)\w*',
    r'\s*,?\s*mantenint el mateix \w+(\s+\w+)?\s*(que|a) l[\'’](original)\w*',
]
 
def clean_description_ca(text: str) -> str:
    """Catalan equivalent of clean_description(). Preserves accents (does
    NOT call normalise_text) — only strips known preamble / original-image
    phrasing and normalises whitespace/punctuation."""
    text = text.strip()
 
    for pat in _PREAMBLE_PATTERNS_CA:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
 
    for pat in _ORIGINAL_REF_PATTERNS_CA:
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
 
    Used by BOTH the English and Catalan scripts: the `query` column is in
    English in both CSVs, so this needs no Catalan variant. In the Catalan
    scripts this still yields the ENGLISH target phrase — see the module
    docstring above for why that's intentional here.
    """
    query = normalise_text(query.strip())
    m = _QUERY_PATTERN.match(query)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", query
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHY PARSING — ENGLISH / TRANSLATED (accents stripped)
#
# PROFESSOR'S REQUIREMENT: "For empty cells in materials_1 and ObjectTypes_1
# — just don't add / do anything (to avoid extra noise)"
#
# Both parse functions return "" (empty string) when the field is empty/NaN.
# An empty string is then silently skipped inside build_messages() so that
# no "Reference image materials: " line appears in Qwen's prompt at all.
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
# HIERARCHY PARSING — CATALAN / ORIGINAL (accents PRESERVED)
#
# Identical $ / | splitting logic to the English versions above, but skips
# normalise_text() so "indumentària", "peça", "camió", "il·luminació" etc.
# keep their correct Catalan spelling instead of being flattened to ASCII.
# Same empty-cell contract: "" in -> "" out -> line omitted from the prompt.
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_first_hierarchy_ca(raw) -> str:
    """Catalan-safe version of parse_first_hierarchy() — accents preserved."""
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    first_group = raw_str.split("$")[0].strip()
    first_term  = first_group.split("|")[0].strip()
    return first_term
 
 
def parse_full_hierarchy_ca(raw) -> str:
    """Catalan-safe version of parse_full_hierarchy() — accents preserved."""
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    groups = [g.strip() for g in raw_str.split("$")]
    seen, terms = set(), []
    for group in groups:
        parts = [p.strip() for p in group.split("|")]
        term  = parts[-1].strip()
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
# PROMPTS — ENGLISH
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
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS — CATALAN
# ─────────────────────────────────────────────────────────────────────────────
 
SYSTEM_PROMPT_CA = (
    "Escrius descripcions visuals breus d'objectes per a un sistema de cerca d'imatges. "
    "IDIOMA: només català. "
    "FORMAT: comença directament amb la descripció de l'objecte. "
    "NO comencis amb 'La imatge mostra', 'La imatge objectiu mostra', "
    "'Es mostra', 'Es representa', ni cap frase semblant. "
    f"LLARGADA: màxim {MAX_WORDS} paraules. Una o dues frases. "
    "CONTINGUT: descriu només propietats visuals — material, textura, color, "
    "acabat de la superfície, forma, estructura. "
    "Fes servir paraules senzilles i habituals en català. Evita termes tècnics o històrics poc habituals. "
    "ESTIL: escriu com si descrivissis el que veus directament davant teu. "
    "Exemple de bona resposta: "
    "'Un mur de maons vermells i rugosos, amb blocs d'argila rectangulars, "
    "juntes de morter profundes i un acabat mat i terrós.' "
    "Exemple de mala resposta: "
    "'La imatge objectiu mostra una superfície de maó que substitueix la ceràmica.' "
    "No esmentis mai l'original, què ha canviat, ni cap transformació."
)
 
STRICT_PROMPT_CA = (
    "Escriu una frase que descrigui un objecte. Només en català. "
    f"Màxim {MAX_WORDS} paraules. "
    "Comença amb 'Un' o 'Una' seguit de la descripció de l'objecte. "
    "Exemple: 'Un bol de porcellana blanca llisa amb una superfície vidriada brillant i forma arrodonida.' "
    "Descriu només: material, color, textura, superfície, forma. "
    "Sense preàmbul. Sense esmentar cap canvi."
)
 
CHANGE_PATTERN_CA = re.compile(
    r"\b(canviat|reemplaçat|reemplacat|substitu[iï]t|transformat|originalment|"
    r"anteriorment|en lloc de|ja no|abans (era|hi havia)|ara (presenta|t[eé])|"
    r"modificat|alterat|ha estat|va ser reemplaçat|va ser canviat|transici[oó])\b",
    re.IGNORECASE,
)
 
 
def _has_non_latin_script(text: str) -> bool:
    """
    Detect CJK, Hiragana/Katakana, Hangul, Cyrillic, Arabic, Hebrew
    characters — a reliable "definitely the wrong language" signal that
    (unlike a plain non-ASCII count) does NOT false-positive on legitimate
    Catalan diacritics (à, è, é, í, ï, ò, ó, ú, ü, ç).
    """
    for c in text:
        cp = ord(c)
        if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
            0x3040 <= cp <= 0x30FF or   # Hiragana / Katakana
            0xAC00 <= cp <= 0xD7A3 or   # Hangul
            0x0400 <= cp <= 0x04FF or   # Cyrillic
            0x0600 <= cp <= 0x06FF or   # Arabic
            0x0590 <= cp <= 0x05FF):    # Hebrew
            return True
    return False
 
 
def build_messages(image_path, mat_str: str, obj_str: str,
                   target_element: str, system: str, lang: str = "en") -> list[dict]:
    """
    Build the Qwen-VL message list for one row.
 
    PROFESSOR'S REQUIREMENT: "For empty cells in materials_1 and ObjectTypes_1
    — just don't add / do anything (to avoid extra noise)"
 
    If mat_str or obj_str is "" (empty — returned by parse_*_hierarchy[_ca]
    when the CSV field is NaN/empty), that context line is simply omitted
    from the prompt.
 
    lang="en" (default) — unchanged English behaviour, identical to before.
    lang="ca" — Catalan-labelled context lines + Catalan instruction text.
                target_element itself is whatever the caller passed in
                (English, from parse_query() on the `query` column — see
                module docstring).
    """
    context_lines: list[str] = []
 
    if lang == "ca":
        if mat_str:
            context_lines.append(f"Materials de la imatge de referència: {mat_str}")
        if obj_str:
            context_lines.append(f"Tipus d'objecte de la imatge de referència: {obj_str}")
    else:
        if mat_str:
            context_lines.append(f"Reference image materials: {mat_str}")
        if obj_str:
            context_lines.append(f"Reference image object types: {obj_str}")
 
    context_block = "\n".join(context_lines)
    if context_block:
        context_block += "\n"
 
    if lang == "ca":
        user_text = (
            f"{context_block}"
            f"Descriu aquest OBJECTIU: {target_element}\n\n"
            f"Escriu una descripció visual breu de {target_element} "
            f"en menys de {MAX_WORDS} paraules. "
            f"Comença directament amb la descripció — sense preàmbul."
        )
    else:
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
                 query: str, id1: str, model, processor, log,
                 lang: str = "en") -> str:
    """
    Generate one target description.
    - Extracts target element from query (not the full transformation)
    - mat_str / obj_str are already parsed by the caller; empty string means
      the field was absent in the CSV — build_messages() will skip it silently
    - Retries with strict prompt if change-language or preamble detected
    - Post-processes to strip any remaining preamble
 
    lang="en" (default): 100% unchanged behaviour vs. the original
                          implementation — existing English scripts calling
                          generate_one() without `lang` are unaffected.
    lang="ca": Catalan prompts/cleaning/change-detection are used instead.
    """
    _, target_element = parse_query(query)
 
    if lang == "ca":
        system_prompt   = SYSTEM_PROMPT_CA
        strict_prompt   = STRICT_PROMPT_CA
        change_pattern  = CHANGE_PATTERN_CA
        cleaner         = clean_description_ca
        max_new_tokens  = MAX_NEW_TOKENS_CA
    else:
        system_prompt   = SYSTEM_PROMPT
        strict_prompt   = STRICT_PROMPT
        change_pattern  = CHANGE_PATTERN
        cleaner         = clean_description
        max_new_tokens  = MAX_NEW_TOKENS
 
    def _run(system: str) -> str:
        messages = build_messages(image_path, mat_str, obj_str, target_element, system, lang=lang)
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
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        n = inputs["input_ids"].shape[1]
        return processor.tokenizer.decode(
            out[0][n:], skip_special_tokens=True).strip()
 
    raw = _run(system_prompt)
 
    # Retry if change-language detected
    if change_pattern.search(raw):
        log.warning(f"  [{id1}] Change-language detected — retrying with strict prompt")
        raw = _run(strict_prompt)
        if change_pattern.search(raw):
            sentences = re.split(r'(?<=[.!?])\s+', raw)
            raw = " ".join(s for s in sentences if not change_pattern.search(s)) or raw
 
    # Retry if wrong-language script detected
    if lang == "ca":
        if _has_non_latin_script(raw):
            log.warning(f"  [{id1}] Non-Latin script detected — retrying with strict prompt")
            raw = _run(strict_prompt)
    else:
        if sum(1 for c in raw if ord(c) > 127) > 3:
            log.warning(f"  [{id1}] Non-English detected — retrying with strict prompt")
            raw = _run(strict_prompt)
 
    # Post-process: strip preamble + original references
    cleaned = cleaner(raw)
 
    # If cleaning made it too short (< 3 words), retry once more
    if len(cleaned.split()) < 3:
        cleaned = cleaner(_run(strict_prompt))
 
    return enforce_word_limit(cleaned, MAX_WORDS)