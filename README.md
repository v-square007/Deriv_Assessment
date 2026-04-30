# Deriv Multilingual Translation Pipeline

A replayable, staged pipeline that fetches Deriv website pages, extracts translatable content, protects brand terms, translates into multiple languages via Google Gemini, reconstructs HTML output, and runs deterministic and LLM-assisted QA.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Validating Output](#validating-output)
- [Pipeline Stages](#pipeline-stages)
- [Output Artifacts](#output-artifacts)
- [Customising Pages and Languages](#customising-pages-and-languages)
- [Efficiency and Caching](#efficiency-and-caching)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| pip | Bundled with Python |
| Google Gemini API key | Free tier at [aistudio.google.com](https://aistudio.google.com/apikey) |

---

## Installation

```bash
# 1. Clone or unzip the repository and enter the directory
cd Deriv_Assessment-main

# 2. Create and activate a virtual environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

`requirements.txt` installs:
- `beautifulsoup4` — HTML parsing and segment extraction
- `requests` — HTTP page fetching
- `google-genai` — Gemini API client (current SDK, not the deprecated `google-generativeai`)
- `lxml` — fast HTML parser backend for BeautifulSoup
- `python-dotenv` — loads `.env` file for API key management

---

## Configuration

### API Key

The pipeline reads your Gemini API key from the environment. The recommended approach is a `.env` file in the project root — this is loaded automatically at startup.

```bash
# Create a .env file in the project root
echo 'GEMINI_API_KEY=your-key-here' > .env
```

Alternatively, export it in your shell session:

```bash
export GEMINI_API_KEY="your-key-here"
```

You can also use `GOOGLE_API_KEY` as an alias — the pipeline accepts either.

**Get a free API key:** https://aistudio.google.com/apikey

### Model Selection (optional)

By default the pipeline uses `gemini-2.0-flash`. To override:

```bash
echo 'GEMINI_MODEL=gemini-1.5-flash' >> .env
```

---

## Running the Pipeline

### Full run (recommended for first use)

```bash
python pipeline.py
```

This fetches all pages in `pages.json`, extracts content, identifies protected terms, translates into every language in `target_languages.json`, reconstructs HTML, runs QA, and writes all output artifacts.

### Skip the HTTP fetch stage

If pages have already been fetched and `extracted_segments.json` exists, you can skip re-fetching to save time and avoid rate limits:

```bash
python pipeline.py --skip-fetch
```

### With LLM-assisted QA (stretch goal)

Adds a Gemini-powered review pass that checks grammar, fluency, tone, and meaning preservation on a sample of translated segments. Uses additional API tokens.

```bash
python pipeline.py --llm-qa
```

### Combined flags

```bash
python pipeline.py --skip-fetch --llm-qa
```

---

## Validating Output

After the pipeline completes, run the validation script to confirm all required artifacts are present and correct:

```bash
python validate.py
```

The validator checks:
- All required JSON and HTML artifacts exist
- JSON files are valid and match expected schemas
- At least 1–2 pages were processed
- Arabic is present when only one language is configured
- Protected terms are unchanged in every translation
- No `[[PROTECTED_N]]` placeholders were left unreplaced
- URLs are preserved in translated output
- Arabic HTML files carry `dir="rtl"` and `lang="ar"` attributes
- QA report was generated and critical issues are surfaced
- `llm_calls.jsonl` contains well-formed records

Exit code `0` means all checks passed (warnings are non-fatal). Exit code `1` means at least one check failed.

---

## Pipeline Stages

The pipeline enforces the following stages in code and prints each transition to the terminal:

```
INIT
  └─> CONFIG_LOADED              pages.json and target_languages.json are read
  └─> PAGES_FETCHED              HTTP fetch via requests + BeautifulSoup
  └─> CONTENT_EXTRACTED          extracted_segments.json is written
  └─> PROTECTED_TERMS_IDENTIFIED protected_terms.json is written
  └─> SEGMENTS_PREPARED          deduplication map is built
  └─> SEGMENTS_DEDUPED_OR_CACHE_CHECKED  translation_cache.json is consulted
  └─> TRANSLATION_COMPLETE       translations/{lang}/ are written
  └─> HTML_RECONSTRUCTED         output/{lang}/*.html are written
  └─> QA_COMPLETE                qa_report.json is written; critical issues printed
  └─> COST_REPORT_GENERATED      cost_report.json and run_metrics.json are written
  └─> RESULTS_FINALISED          summary is printed to terminal
```

The pipeline exits with code `1` if any critical QA issues are found.

---

## Output Artifacts

All artifacts are written to the project root unless noted.

| Artifact | Description |
|---|---|
| `extracted_segments.json` | All translatable segments with HTML path, source text, links, and placeholders |
| `protected_terms.json` | Brand and product terms excluded from translation |
| `translations/{lang}/translated_segments.json` | Per-language translated segments with QA status |
| `output/{lang}/*.html` | Reconstructed HTML pages; Arabic output includes `dir="rtl"` |
| `qa_report.json` | Deterministic QA results with severity levels |
| `llm_qa_report.json` | LLM-assisted QA results (`--llm-qa` only) |
| `cost_report.json` | Token usage and estimated USD cost per language and page |
| `translation_cache.json` | Persistent translation cache keyed by content hash |
| `run_metrics.json` | Stage timings, cache hit rate, QA pass rate |
| `llm_calls.jsonl` | Append-only log of every Gemini API call with prompt hash and token estimates |
| `fetch_failures.json` | Pages that could not be fetched (non-fatal) |

---

## Customising Pages and Languages

### Change the pages to translate

Edit `pages.json`:

```json
{
  "pages": [
    "https://deriv.com/",
    "https://deriv.com/markets/forex/",
    "https://deriv.com/regulatory/"
  ]
}
```

### Change the target languages

Edit `target_languages.json`:

```json
{
  "target_languages": [
    { "code": "ar", "name": "Arabic",          "direction": "rtl" },
    { "code": "ms", "name": "Bahasa Malaysia", "direction": "ltr" },
    { "code": "pt", "name": "Portuguese",      "direction": "ltr" }
  ]
}
```

> **Note:** Arabic (`ar`) is mandatory. If it is absent from `target_languages.json`, the pipeline will add it automatically.

The evaluator may replace either config file before running the pipeline. The pipeline reads both files fresh on every run — no code changes are needed.

---

## Efficiency and Caching

The pipeline applies two strategies to avoid redundant LLM calls:

**In-run deduplication** — segments with identical content (after protected-term substitution) are translated once per language per run, regardless of how many pages they appear on.

**Persistent cache** — `translation_cache.json` stores results keyed by `(SHA-256(content), language_code, SHA-256(protected_terms_list))`. On subsequent runs, cached translations are reused and the cache hit rate is reported in `run_metrics.json`.

To force a full re-translation (e.g. after changing the prompt or protected terms), delete `translation_cache.json` before running.

---

## Troubleshooting

### `EnvironmentError: Set GEMINI_API_KEY`
The API key is missing. Create a `.env` file in the project root:
```bash
echo 'GEMINI_API_KEY=your-key-here' > .env
```

### Pages appear in `fetch_failures.json`
Deriv pages may block automated scrapers or return JavaScript-rendered content that requires a headless browser. The pipeline continues with whichever segments were successfully extracted. Use `--skip-fetch` on re-runs to avoid re-attempting failed pages.

### `ModuleNotFoundError`
Your virtual environment may not be activated, or dependencies may not be installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `FutureWarning` about `google.generativeai`
This project uses the current `google-genai` SDK. If you see this warning, an old version of the SDK may be installed alongside the new one. Uninstall the deprecated package:
```bash
pip uninstall google-generativeai -y
```

### Validation fails with RTL warnings
Arabic output HTML must include `dir="rtl"`. This is written automatically by the pipeline. If it is missing, re-run the pipeline without `--skip-fetch` to regenerate the HTML output files.
