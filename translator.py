"""
translator.py
-------------
Stages: PROTECTED_TERMS_IDENTIFIED -> SEGMENTS_PREPARED ->
        SEGMENTS_DEDUPED_OR_CACHE_CHECKED -> TRANSLATION_COMPLETE ->
        HTML_RECONSTRUCTED

Responsibilities:
  - Infer and save protected_terms.json from segment content
  - Replace protected terms with reversible [[PROTECTED_N]] placeholders
  - Deduplicate segments by content hash before calling the LLM
  - Translate via Google Gemini 1.5 Flash
  - Restore protected terms in translated output
  - Inject RTL metadata for Arabic
  - Log every LLM call to llm_calls.jsonl
  - Cache translations in translation_cache.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

EXTRACTED_SEGMENTS_JSON = ROOT / "extracted_segments.json"
PROTECTED_TERMS_JSON = ROOT / "protected_terms.json"
TRANSLATION_CACHE_JSON = ROOT / "translation_cache.json"
LLM_CALLS_JSONL = ROOT / "llm_calls.jsonl"
TRANSLATIONS_DIR = ROOT / "translations"
OUTPUT_DIR = ROOT / "output"

# Gemini model to use
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Tokens are approximated as len(text) / 4  (rough heuristic)
CHARS_PER_TOKEN = 4

# Seconds to wait between API calls to avoid rate-limit bursts
API_CALL_DELAY = 0.5

# ---------------------------------------------------------------------------
# Brand / product terms that must NEVER be translated.
# The pipeline also infers additional terms from page content.
# ---------------------------------------------------------------------------

SEED_PROTECTED_TERMS: list[str] = [
    "Deriv",
    "Deriv Bot",
    "Deriv MT5",
    "Deriv X",
    "Deriv GO",
    "Deriv EZ",
    "SmartTrader",
    "DTrader",
    "DBot",
    "DMT5",
    "DerivX",
    "CFD",
    "CFDs",
    "Forex",
    "Synthetic Indices",
    "Multipliers",
    "Accumulators",
    "Turbos",
    "Vanilla Options",
    "MT5",
    "MetaTrader 5",
    "MetaTrader 4",
    "MT4",
    "TradingView",
    "cTrader",
    "Options & Multipliers",
]

# Regex to detect additional product-like terms in scraped text
INFER_TERM_PATTERN = re.compile(
    r"\b(Deriv\w*|SmartTrader|DBot|DTrader|DMT5|DerivX|MT5|MT4"
    r"|CFDs?|Forex|Synthetic\s+Indices?|Multipliers?|Accumulators?"
    r"|Turbos?|Vanilla\s+Options?|TradingView|cTrader)\b"
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _append_llm_call(record: dict[str, Any]) -> None:
    with LLM_CALLS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _get_gemini_client() -> "genai.Client":
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Protected-term inference
# ---------------------------------------------------------------------------


def infer_protected_terms(segments: list[dict[str, Any]]) -> list[str]:
    """
    Combine seed list with any additional brand/product terms found in segments.
    Returns a deduplicated, longest-first sorted list (important for placeholder
    substitution so 'Deriv Bot' is matched before 'Deriv').
    """
    found: set[str] = set(SEED_PROTECTED_TERMS)

    for seg in segments:
        text = seg.get("plain_text", "") or seg.get("source_text", "")
        for m in INFER_TERM_PATTERN.finditer(text):
            found.add(m.group(0).strip())

    # Sort longest first so multi-word terms match before sub-terms
    return sorted(found, key=lambda t: -len(t))


def save_protected_terms(terms: list[str], path: Path = PROTECTED_TERMS_JSON) -> None:
    data = {"protected_terms": terms, "count": len(terms)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(terms)} protected terms -> {path}")


def load_protected_terms(path: Path = PROTECTED_TERMS_JSON) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["protected_terms"]


# ---------------------------------------------------------------------------
# Placeholder protection / restoration
# ---------------------------------------------------------------------------


def protect_text(text: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    """
    Replace each protected term with [[PROTECTED_N]].
    Returns (protected_text, mapping) where mapping[placeholder] = original_term.
    Terms must already be sorted longest-first.
    """
    mapping: dict[str, str] = {}
    counter = 0

    # Escape special regex chars in terms
    for term in terms:
        pattern = re.compile(r"(?<!\[\[)" + re.escape(term) + r"(?!\]\])", re.IGNORECASE)
        placeholder = f"[[PROTECTED_{counter}]]"

        def replacer(m: re.Match, ph: str = placeholder, orig_term: str = term) -> str:  # noqa: E501
            return ph

        new_text, n = pattern.subn(replacer, text)
        if n > 0:
            mapping[placeholder] = term
            text = new_text
            counter += 1

    return text, mapping


def restore_text(text: str, mapping: dict[str, str]) -> str:
    """Reverse the placeholder substitution."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


# ---------------------------------------------------------------------------
# Deduplication / cache
# ---------------------------------------------------------------------------


def load_cache(path: Path = TRANSLATION_CACHE_JSON) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"entries": {}, "hits": 0, "misses": 0}


def save_cache(cache: dict[str, Any], path: Path = TRANSLATION_CACHE_JSON) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(protected_text: str, lang_code: str, terms_hash: str) -> str:
    raw = f"{protected_text}|{lang_code}|{terms_hash}"
    return _sha256(raw)


# ---------------------------------------------------------------------------
# LLM translation call
# ---------------------------------------------------------------------------

_TRANSLATION_PROMPT_TEMPLATE = """\
You are a professional translator.

Translate the following HTML/text content from English into {lang_name} ({lang_code}).

RULES — you must follow ALL of these exactly:
1. Preserve every HTML tag unchanged (e.g. <strong>, <em>, <a href="...">, etc.).
2. Preserve every placeholder exactly as-is: [[PROTECTED_N]] tokens must not be changed,
   translated, or moved. They represent brand/product names.
3. Preserve all URLs, href values, and src attributes unchanged.
4. Use natural, fluent {lang_name} suitable for a professional financial trading website.
5. Maintain the original tone (professional, direct, marketing-oriented).
6. Do NOT add any explanation, commentary, or extra text — output ONLY the translated content.
7. Do NOT wrap the output in markdown code fences.
{rtl_instruction}

Source content:
{source_text}
"""

_RTL_INSTRUCTION = (
    "8. This is a RIGHT-TO-LEFT language. Ensure the translation reads naturally "
    "in RTL context. Do not add dir or lang attributes — those are handled separately."
)


def _build_prompt(
    source_text: str,
    lang_code: str,
    lang_name: str,
    is_rtl: bool,
) -> str:
    rtl_instruction = _RTL_INSTRUCTION if is_rtl else ""
    return _TRANSLATION_PROMPT_TEMPLATE.format(
        lang_name=lang_name,
        lang_code=lang_code,
        source_text=source_text,
        rtl_instruction=rtl_instruction,
    )


def translate_text(
    client: "genai.Client",
    source_text: str,
    lang_code: str,
    lang_name: str,
    is_rtl: bool,
    page_url: str | None,
    segment_id: str,
) -> tuple[str, dict[str, Any]]:
    """
    Call Gemini to translate *source_text*.
    Returns (translated_text, llm_call_record).
    """
    prompt = _build_prompt(source_text, lang_code, lang_name, is_rtl)
    prompt_hash = _sha256(prompt)

    est_input = _estimate_tokens(prompt)
    est_output = _estimate_tokens(source_text) * 2  # rough upper bound

    call_record: dict[str, Any] = {
        "stage": "TRANSLATION",
        "page_url": page_url,
        "language_code": lang_code,
        "segment_id": segment_id,
        "timestamp": _now_iso(),
        "provider": "google",
        "model": GEMINI_MODEL,
        "prompt_hash": prompt_hash,
        "input_artifacts": [str(EXTRACTED_SEGMENTS_JSON), str(PROTECTED_TERMS_JSON)],
        "output_artifact": str(TRANSLATIONS_DIR / lang_code),
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "status": "attempted",
    }

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        translated = response.text.strip()

        # Update token estimate from usage metadata if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            call_record["estimated_input_tokens"] = getattr(meta, "prompt_token_count", est_input)
            call_record["estimated_output_tokens"] = getattr(meta, "candidates_token_count", est_output)

        call_record["status"] = "success"
        _append_llm_call(call_record)
        return translated, call_record
    except Exception as exc:  # noqa: BLE001
        call_record["status"] = "error"
        call_record["error"] = str(exc)[:500]
        _append_llm_call(call_record)
        raise


# ---------------------------------------------------------------------------
# AI-assisted protected-term identification (logged separately)
# ---------------------------------------------------------------------------


def ai_identify_protected_terms(
    client: "genai.Client",
    sample_text: str,
) -> list[str]:
    """
    Ask Gemini to identify additional brand/product terms in sample_text.
    Returns a list of terms to add to the protected list.
    """
    prompt = (
        "You are analysing content from a financial trading website (Deriv).\n"
        "List every brand name, product name, platform name, or technical term "
        "that should remain untranslated into any language.\n"
        "Return ONLY a JSON array of strings. No explanation, no markdown fences.\n\n"
        f"Content:\n{sample_text[:3000]}"
    )
    prompt_hash = _sha256(prompt)

    call_record: dict[str, Any] = {
        "stage": "PROTECTED_TERMS_IDENTIFICATION",
        "page_url": None,
        "language_code": None,
        "timestamp": _now_iso(),
        "provider": "google",
        "model": GEMINI_MODEL,
        "prompt_hash": prompt_hash,
        "input_artifacts": [str(EXTRACTED_SEGMENTS_JSON)],
        "output_artifact": str(PROTECTED_TERMS_JSON),
        "estimated_input_tokens": _estimate_tokens(prompt),
        "estimated_output_tokens": 200,
    }

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        terms = json.loads(raw)
        if isinstance(terms, list):
            _append_llm_call(call_record)
            return [str(t) for t in terms if isinstance(t, str)]
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] AI term identification failed: {exc}")

    _append_llm_call(call_record)
    return []


# ---------------------------------------------------------------------------
# HTML reconstruction
# ---------------------------------------------------------------------------


def inject_rtl_metadata(html_content: str, lang_code: str, lang_name: str) -> str:
    """
    Wrap translated content in a minimal HTML shell with RTL metadata for Arabic.
    """
    is_rtl = lang_code == "ar"
    direction = "rtl" if is_rtl else "ltr"

    return f"""<!DOCTYPE html>
<html lang="{lang_code}" dir="{direction}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Deriv – {lang_name}</title>
  {"<style>body { direction: rtl; text-align: right; }</style>" if is_rtl else ""}
</head>
<body>
{html_content}
</body>
</html>"""


def reconstruct_html_for_page(
    page_url: str,
    translated_segments: list[dict[str, Any]],
    lang_code: str,
    lang_name: str,
) -> str:
    """Build a simple HTML page from translated segments."""
    parts: list[str] = [f"<!-- Source: {page_url} -->\n"]
    for seg in translated_segments:
        parts.append(seg.get("translated_text", "") + "\n")

    body_content = "\n".join(parts)
    return inject_rtl_metadata(body_content, lang_code, lang_name)


# ---------------------------------------------------------------------------
# Main translation orchestrator
# ---------------------------------------------------------------------------


def run_translation(
    segments: list[dict[str, Any]],
    languages: list[dict[str, Any]],
    protected_terms: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """
    Translate all segments into each target language.

    Returns dict[lang_code -> list[translated_segment_dict]].
    """
    client = _get_gemini_client()

    # Build a stable hash of the protected terms list (for cache keying)
    terms_hash = _sha256(json.dumps(protected_terms, sort_keys=True))

    cache = load_cache()
    results: dict[str, list[dict[str, Any]]] = {lang["code"]: [] for lang in languages}

    # Deduplication map: content_hash -> first segment that had this text
    dedup_map: dict[str, str] = {}  # hash -> segment_id
    # Translations already done this run (content_hash, lang) -> translated_text
    run_translation_memo: dict[tuple[str, str], str] = {}

    for lang in languages:
        lang_code: str = lang["code"]
        lang_name: str = lang["name"]
        is_rtl: bool = lang.get("direction", "ltr") == "rtl"

        print(f"\n  Translating into {lang_name} ({lang_code})…")

        lang_dir = TRANSLATIONS_DIR / lang_code
        lang_dir.mkdir(parents=True, exist_ok=True)

        translated_list: list[dict[str, Any]] = []

        for seg in segments:
            segment_id: str = seg["segment_id"]
            source_text: str = seg.get("source_text", "")
            page_url: str = seg.get("page_url", "")

            if not source_text.strip():
                continue

            # --- Protect terms ---
            protected_text, mapping = protect_text(source_text, protected_terms)

            # --- Deduplication via content hash ---
            content_hash = _sha256(protected_text)

            cache_k = _cache_key(protected_text, lang_code, terms_hash)
            translated_text: str | None = None

            # 1. Check in-run memo (handles exact duplicates within same run)
            memo_key = (content_hash, lang_code)
            if memo_key in run_translation_memo:
                translated_text = run_translation_memo[memo_key]
                cache["hits"] = cache.get("hits", 0) + 1
                print(f"    [DEDUP] {segment_id[:8]}… (in-run duplicate)")

            # 2. Check persistent cache
            elif cache_k in cache.get("entries", {}):
                entry = cache["entries"][cache_k]
                translated_text = entry["translated_text"]
                cache["hits"] = cache.get("hits", 0) + 1
                print(f"    [CACHE] {segment_id[:8]}… (cache hit)")

            # 3. Call LLM
            else:
                cache["misses"] = cache.get("misses", 0) + 1
                try:
                    translated_text, _ = translate_text(
                        client,
                        protected_text,
                        lang_code,
                        lang_name,
                        is_rtl,
                        page_url,
                        segment_id,
                    )
                    # Store in cache
                    cache.setdefault("entries", {})[cache_k] = {
                        "translated_text": translated_text,
                        "lang_code": lang_code,
                        "content_hash": content_hash,
                        "terms_hash": terms_hash,
                        "created_at": _now_iso(),
                    }
                    run_translation_memo[memo_key] = translated_text
                    print(f"    [API]   {segment_id[:8]}… translated")
                    time.sleep(API_CALL_DELAY)

                except Exception as exc:  # noqa: BLE001
                    print(f"    [ERROR] {segment_id}: {exc}")
                    translated_text = source_text  # fallback: keep source

            # --- Restore protected terms ---
            final_text = restore_text(translated_text, mapping)

            record: dict[str, Any] = {
                "segment_id": segment_id,
                "language_code": lang_code,
                "page_url": page_url,
                "source_text": source_text,
                "protected_text": protected_text,
                "translated_text": final_text,
                "protected_terms_restored": True,
                "content_hash": content_hash,
                "qa_status": "pending",
            }
            translated_list.append(record)

        results[lang_code] = translated_list

        # Persist per-language translations JSON
        out_path = lang_dir / "translated_segments.json"
        out_path.write_text(
            json.dumps(translated_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  -> Saved {len(translated_list)} translations -> {out_path}")

    # Persist updated cache
    save_cache(cache)
    print(f"\n  Cache: {cache.get('hits', 0)} hits / {cache.get('misses', 0)} misses")

    return results


# ---------------------------------------------------------------------------
# HTML output reconstruction
# ---------------------------------------------------------------------------


def run_reconstruction(
    segments: list[dict[str, Any]],
    translated_results: dict[str, list[dict[str, Any]]],
    languages: list[dict[str, Any]],
) -> None:
    """Write reconstructed HTML output files per language, grouped by page."""
    for lang in languages:
        lang_code = lang["code"]
        lang_name = lang["name"]
        translated_segs = translated_results.get(lang_code, [])

        # Group by page_url
        pages: dict[str, list[dict[str, Any]]] = {}
        for seg in translated_segs:
            pages.setdefault(seg["page_url"], []).append(seg)

        out_lang_dir = OUTPUT_DIR / lang_code
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        for page_url, page_segs in pages.items():
            # Create a safe filename from the URL
            safe_name = re.sub(r"[^\w]", "_", page_url.replace("https://", ""))[:80]
            out_file = out_lang_dir / f"{safe_name}.html"

            html = reconstruct_html_for_page(page_url, page_segs, lang_code, lang_name)
            out_file.write_text(html, encoding="utf-8")
            print(f"  -> HTML output: {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    segs = json.loads(EXTRACTED_SEGMENTS_JSON.read_text(encoding="utf-8"))
    langs_data = json.loads((ROOT / "target_languages.json").read_text(encoding="utf-8"))
    langs = langs_data["target_languages"]

    terms = infer_protected_terms(segs)
    save_protected_terms(terms)

    results = run_translation(segs, langs, terms)
    run_reconstruction(segs, results, langs)
    print("Translation done.")