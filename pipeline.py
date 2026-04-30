"""
pipeline.py
-----------
Main pipeline orchestrator.

Stages enforced in code:
  INIT -> CONFIG_LOADED -> PAGES_FETCHED -> CONTENT_EXTRACTED ->
  PROTECTED_TERMS_IDENTIFIED -> SEGMENTS_PREPARED ->
  SEGMENTS_DEDUPED_OR_CACHE_CHECKED -> TRANSLATION_COMPLETE ->
  HTML_RECONSTRUCTED -> QA_COMPLETE -> COST_REPORT_GENERATED ->
  RESULTS_FINALISED

Run:
    python pipeline.py [--llm-qa] [--skip-fetch]

Options:
  --llm-qa       Run optional LLM-assisted QA (costs extra tokens)
  --skip-fetch   Re-use existing extracted_segments.json (skip fetch stage)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Stage enum / tracker
# ---------------------------------------------------------------------------

STAGES = [
    "INIT",
    "CONFIG_LOADED",
    "PAGES_FETCHED",
    "CONTENT_EXTRACTED",
    "PROTECTED_TERMS_IDENTIFIED",
    "SEGMENTS_PREPARED",
    "SEGMENTS_DEDUPED_OR_CACHE_CHECKED",
    "TRANSLATION_COMPLETE",
    "HTML_RECONSTRUCTED",
    "QA_COMPLETE",
    "COST_REPORT_GENERATED",
    "RESULTS_FINALISED",
]


class PipelineState:
    def __init__(self) -> None:
        self.current: str = "INIT"
        self.timings: dict[str, float] = {}
        self._start: float = time.perf_counter()

    def advance(self, stage: str) -> None:
        elapsed = time.perf_counter() - self._start
        self.timings[stage] = round(elapsed, 3)
        self.current = stage
        print(f"\n{'='*60}")
        print(f"  STAGE: {stage}  (+{elapsed:.2f}s)")
        print(f"{'='*60}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_stage": self.current,
            "stage_timings_sec": self.timings,
        }


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


def build_cost_report(
    llm_calls: list[dict[str, Any]],
    languages: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    cache_hits: int,
    cache_misses: int,
) -> dict[str, Any]:
    """
    Estimate cost from logged LLM calls.
    Gemini 1.5 Flash pricing (as of 2025):
      Input:  $0.075 / 1M tokens
      Output: $0.30  / 1M tokens
    """
    INPUT_COST_PER_M = 0.075
    OUTPUT_COST_PER_M = 0.30

    total_input_tokens = sum(c.get("estimated_input_tokens", 0) for c in llm_calls)
    total_output_tokens = sum(c.get("estimated_output_tokens", 0) for c in llm_calls)

    total_cost = (
        total_input_tokens / 1_000_000 * INPUT_COST_PER_M
        + total_output_tokens / 1_000_000 * OUTPUT_COST_PER_M
    )

    # Per-language breakdown
    per_language: dict[str, Any] = {}
    for lang in languages:
        code = lang["code"]
        lang_calls = [c for c in llm_calls if c.get("language_code") == code]
        inp = sum(c.get("estimated_input_tokens", 0) for c in lang_calls)
        out = sum(c.get("estimated_output_tokens", 0) for c in lang_calls)
        cost = inp / 1_000_000 * INPUT_COST_PER_M + out / 1_000_000 * OUTPUT_COST_PER_M
        per_language[code] = {
            "calls": len(lang_calls),
            "estimated_input_tokens": inp,
            "estimated_output_tokens": out,
            "estimated_cost_usd": round(cost, 6),
        }

    # Per-page breakdown
    per_page: dict[str, Any] = {}
    for seg in segments:
        url = seg.get("page_url", "unknown")
        per_page.setdefault(url, {"segments": 0})
        per_page[url]["segments"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-1.5-flash",
        "pricing_note": "Gemini 1.5 Flash: $0.075/1M input tokens, $0.30/1M output tokens",
        "efficiency_strategy": (
            "Segment deduplication by SHA-256 content hash prevents re-translating identical "
            "segments across pages. A persistent translation_cache.json stores results by "
            "(content_hash, lang_code, terms_hash) so repeat pipeline runs reuse prior work."
        ),
        "cache_summary": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate_pct": round(cache_hits / max(1, cache_hits + cache_misses) * 100, 1),
        },
        "totals": {
            "llm_calls": len(llm_calls),
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_total_cost_usd": round(total_cost, 6),
        },
        "per_language": per_language,
        "per_page_segments": per_page,
    }


# ---------------------------------------------------------------------------
# Run metrics
# ---------------------------------------------------------------------------


def build_run_metrics(
    state: PipelineState,
    segments: list[dict[str, Any]],
    translated_results: dict[str, list[dict[str, Any]]],
    qa_issues: list[dict[str, Any]],
    cache: dict[str, Any],
    llm_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    total_segs = len(segments)
    hits = cache.get("hits", 0)
    misses = cache.get("misses", 0)

    per_lang: dict[str, Any] = {}
    for lang_code, segs in translated_results.items():
        critical = sum(
            1 for i in qa_issues
            if i["language_code"] == lang_code and i["severity"] == "critical"
        )
        per_lang[lang_code] = {
            "translated_segments": len(segs),
            "critical_qa_issues": critical,
            "qa_pass_rate_pct": round(
                (len(segs) - critical) / max(1, len(segs)) * 100, 1
            ),
        }

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "stage_timings_sec": state.timings,
        "segments_extracted": total_segs,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate_pct": round(hits / max(1, hits + misses) * 100, 1),
        "total_llm_calls": len(llm_calls),
        "total_qa_issues": len(qa_issues),
        "per_language": per_lang,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def load_llm_calls() -> list[dict[str, Any]]:
    path = ROOT / "llm_calls.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def run_pipeline(enable_llm_qa: bool = False, skip_fetch: bool = False) -> None:
    state = PipelineState()

    # Clear llm_calls.jsonl at start of fresh run
    llm_calls_path = ROOT / "llm_calls.jsonl"
    if llm_calls_path.exists() and not skip_fetch:
        llm_calls_path.unlink()
    llm_calls_path.touch()

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    state.advance("INIT")
    from ingest import load_pages, EXTRACTED_SEGMENTS_JSON, PAGES_JSON
    from translator import (
        infer_protected_terms,
        save_protected_terms,
        load_protected_terms,
        run_translation,
        run_reconstruction,
        ai_identify_protected_terms,
        load_cache,
        PROTECTED_TERMS_JSON,
        TRANSLATION_CACHE_JSON,
        TRANSLATIONS_DIR,
        OUTPUT_DIR,
    )
    from qa import (
        run_deterministic_qa,
        save_qa_report,
        run_llm_qa,
        save_llm_qa_report,
    )

    TRANSLATIONS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # CONFIG_LOADED
    # ------------------------------------------------------------------
    state.advance("CONFIG_LOADED")
    pages = load_pages(ROOT / "pages.json")
    langs_data = json.loads((ROOT / "target_languages.json").read_text(encoding="utf-8"))
    languages: list[dict[str, Any]] = langs_data["target_languages"]

    print(f"  Pages   : {len(pages)}")
    print(f"  Languages: {[l['code'] for l in languages]}")

    # Ensure Arabic is always present
    lang_codes = [l["code"] for l in languages]
    if "ar" not in lang_codes:
        print("  [WARN] Arabic not in target_languages.json — adding it (required).")
        languages.insert(
            0,
            {"code": "ar", "name": "Arabic", "direction": "rtl"},
        )

    # ------------------------------------------------------------------
    # PAGES_FETCHED  (or skipped)
    # ------------------------------------------------------------------
    if skip_fetch and EXTRACTED_SEGMENTS_JSON.exists():
        state.advance("PAGES_FETCHED")
        print("  [SKIP] Using cached extracted_segments.json")
    else:
        state.advance("PAGES_FETCHED")
        from ingest import run_ingest, save_segments
        segments = run_ingest(pages)

        # ------------------------------------------------------------------
        # CONTENT_EXTRACTED
        # ------------------------------------------------------------------
        state.advance("CONTENT_EXTRACTED")
        save_segments(segments, EXTRACTED_SEGMENTS_JSON)

    # Load segments from disk (canonical source)
    segments: list[dict[str, Any]] = json.loads(
        EXTRACTED_SEGMENTS_JSON.read_text(encoding="utf-8")
    )
    if not skip_fetch:
        pass  # already advanced to CONTENT_EXTRACTED above
    else:
        state.advance("CONTENT_EXTRACTED")
        print(f"  Loaded {len(segments)} segments from disk")

    # ------------------------------------------------------------------
    # PROTECTED_TERMS_IDENTIFIED
    # ------------------------------------------------------------------
    state.advance("PROTECTED_TERMS_IDENTIFIED")

    # Try AI-assisted identification on a sample of text
    sample_text = " ".join(
        s.get("plain_text", "") for s in segments[:30]
    )

    try:
        from translator import _get_gemini_client
        client = _get_gemini_client()
        ai_terms = ai_identify_protected_terms(client, sample_text)
        print(f"  AI identified {len(ai_terms)} additional terms")
    except Exception as exc:  # noqa: BLE001
        ai_terms = []
        print(f"  [WARN] AI term identification skipped: {exc}")

    all_terms = infer_protected_terms(segments)
    # Merge AI terms
    for t in ai_terms:
        if t not in all_terms:
            all_terms.append(t)
    # Re-sort longest first
    all_terms = sorted(set(all_terms), key=lambda x: -len(x))
    save_protected_terms(all_terms, PROTECTED_TERMS_JSON)

    # ------------------------------------------------------------------
    # SEGMENTS_PREPARED
    # ------------------------------------------------------------------
    state.advance("SEGMENTS_PREPARED")
    print(f"  {len(segments)} segments ready for translation")

    # ------------------------------------------------------------------
    # SEGMENTS_DEDUPED_OR_CACHE_CHECKED  (handled inside run_translation)
    # ------------------------------------------------------------------
    state.advance("SEGMENTS_DEDUPED_OR_CACHE_CHECKED")
    print("  Deduplication and cache check will occur during translation")

    # ------------------------------------------------------------------
    # TRANSLATION_COMPLETE
    # ------------------------------------------------------------------
    state.advance("TRANSLATION_COMPLETE")
    protected_terms = load_protected_terms(PROTECTED_TERMS_JSON)
    translated_results = run_translation(segments, languages, protected_terms)

    # ------------------------------------------------------------------
    # HTML_RECONSTRUCTED
    # ------------------------------------------------------------------
    state.advance("HTML_RECONSTRUCTED")
    run_reconstruction(segments, translated_results, languages)

    # ------------------------------------------------------------------
    # QA_COMPLETE
    # ------------------------------------------------------------------
    state.advance("QA_COMPLETE")
    qa_issues = run_deterministic_qa(translated_results, protected_terms)
    save_qa_report(qa_issues, translated_results)

    if enable_llm_qa:
        print("\n  Running LLM-assisted QA…")
        llm_qa_issues = run_llm_qa(translated_results, languages)
        save_llm_qa_report(llm_qa_issues)

    # ------------------------------------------------------------------
    # COST_REPORT_GENERATED
    # ------------------------------------------------------------------
    state.advance("COST_REPORT_GENERATED")
    llm_calls = load_llm_calls()
    cache = load_cache(TRANSLATION_CACHE_JSON)
    cost_report = build_cost_report(
        llm_calls,
        languages,
        segments,
        cache.get("hits", 0),
        cache.get("misses", 0),
    )
    cost_path = ROOT / "cost_report.json"
    cost_path.write_text(json.dumps(cost_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Cost report saved -> {cost_path}")
    print(f"  Estimated total cost: ${cost_report['totals']['estimated_total_cost_usd']:.4f} USD")

    # ------------------------------------------------------------------
    # RESULTS_FINALISED
    # ------------------------------------------------------------------
    state.advance("RESULTS_FINALISED")

    # Run metrics
    metrics = build_run_metrics(state, segments, translated_results, qa_issues, cache, llm_calls)
    metrics_path = ROOT / "run_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Run metrics saved -> {metrics_path}")

    # Summary
    total_translated = sum(len(v) for v in translated_results.values())
    critical_issues = sum(1 for i in qa_issues if i["severity"] == "critical")
    warnings = sum(1 for i in qa_issues if i["severity"] == "warning")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Segments extracted  : {len(segments)}")
    print(f"  Segments translated : {total_translated}")
    print(f"  Languages           : {[l['code'] for l in languages]}")
    print(f"  QA issues (critical): {critical_issues}")
    print(f"  QA issues (warning) : {warnings}")
    print(f"  LLM calls logged    : {len(llm_calls)}")
    print(f"  Cache hits/misses   : {cache.get('hits',0)} / {cache.get('misses',0)}")
    print(f"  Est. cost           : ${cost_report['totals']['estimated_total_cost_usd']:.4f} USD")

    if critical_issues > 0:
        print(f"\n  *** WARNING: {critical_issues} CRITICAL QA issues require review ***")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deriv multilingual translation pipeline")
    parser.add_argument(
        "--llm-qa",
        action="store_true",
        default=False,
        help="Enable optional LLM-assisted QA (uses additional tokens)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        default=False,
        help="Re-use existing extracted_segments.json (skip HTTP fetch stage)",
    )
    args = parser.parse_args()
    run_pipeline(enable_llm_qa=args.llm_qa, skip_fetch=args.skip_fetch)