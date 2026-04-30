"""
qa.py
-----
Stage: QA_COMPLETE

Runs deterministic QA checks on every translated segment.
Optionally runs LLM-assisted QA (grammar, fluency, tone).

Saves:
  qa_report.json
  llm_qa_report.json  (if LLM QA is enabled)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

QA_REPORT_JSON = ROOT / "qa_report.json"
LLM_QA_REPORT_JSON = ROOT / "llm_qa_report.json"
LLM_CALLS_JSONL = ROOT / "llm_calls.jsonl"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _append_llm_call(record: dict[str, Any]) -> None:
    with LLM_CALLS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


PROTECTED_PLACEHOLDER_RE = re.compile(r"\[\[PROTECTED_\d+\]\]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r'https?://[^\s"\'<>]+')


# ---------------------------------------------------------------------------
# Deterministic QA checks
# ---------------------------------------------------------------------------


def _check_empty(seg: dict[str, Any]) -> dict[str, Any] | None:
    if not seg.get("translated_text", "").strip():
        return {
            "issue_type": "empty_translation",
            "severity": "critical",
            "details": "Translated text is empty.",
        }
    return None


def _check_untranslated(seg: dict[str, Any]) -> dict[str, Any] | None:
    src = seg.get("source_text", "").strip()
    trn = seg.get("translated_text", "").strip()
    if src and trn and src == trn:
        # Allow if it's very short or looks like a brand-only string
        if len(src) > 30:
            return {
                "issue_type": "untranslated_segment",
                "severity": "warning",
                "details": "Translated text is identical to source text.",
            }
    return None


def _check_placeholder_corruption(seg: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    src = seg.get("source_text", "")
    trn = seg.get("translated_text", "")

    # Detect leftover [[PROTECTED_N]] in translated output
    # (should have been restored before saving)
    leftovers = PROTECTED_PLACEHOLDER_RE.findall(trn)
    if leftovers:
        issues.append(
            {
                "issue_type": "placeholder_not_restored",
                "severity": "critical",
                "details": f"Unreplaced placeholders found: {leftovers}",
            }
        )

    # Check that segment-declared placeholders still appear
    declared = seg.get("placeholders", [])
    for ph in declared:
        if ph not in trn:
            issues.append(
                {
                    "issue_type": "placeholder_missing",
                    "severity": "critical",
                    "details": f"Placeholder '{ph}' from source is absent in translation.",
                }
            )

    return issues


def _check_url_preservation(seg: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    src_urls = set(URL_RE.findall(seg.get("source_text", "")))
    trn_urls = set(URL_RE.findall(seg.get("translated_text", "")))
    for url in src_urls:
        if url not in trn_urls:
            issues.append(
                {
                    "issue_type": "url_missing",
                    "severity": "critical",
                    "details": f"URL lost in translation: {url}",
                }
            )
    return issues


def _check_broken_html(seg: dict[str, Any]) -> dict[str, Any] | None:
    trn = seg.get("translated_text", "")
    # Simple unbalanced tag heuristic
    opens = re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*/?>", trn)
    closes = re.findall(r"</([a-zA-Z][a-zA-Z0-9]*)>", trn)
    open_counts: dict[str, int] = {}
    for tag in opens:
        open_counts[tag] = open_counts.get(tag, 0) + 1
    for tag in closes:
        open_counts[tag] = open_counts.get(tag, 0) - 1
    broken = [t for t, c in open_counts.items() if c != 0 and t not in {"br", "hr", "img", "input", "meta", "link"}]
    if broken:
        return {
            "issue_type": "broken_html_tags",
            "severity": "warning",
            "details": f"Possibly unbalanced tags: {broken}",
        }
    return None


def _check_rtl(seg: dict[str, Any], lang_code: str) -> dict[str, Any] | None:
    """For Arabic, check whether the output HTML (if any) has dir=rtl."""
    if lang_code != "ar":
        return None

    # We check the reconstructed HTML file, not the segment text.
    # This check is deferred to validate.py for file-level checks.
    # Here we flag if the translated_text itself contains an html tag without dir=rtl.
    trn = seg.get("translated_text", "")
    html_match = re.search(r"<html([^>]*)>", trn, re.IGNORECASE)
    if html_match:
        attrs = html_match.group(1)
        if 'dir="rtl"' not in attrs and "dir='rtl'" not in attrs:
            return {
                "issue_type": "rtl_direction_missing",
                "severity": "critical",
                "details": "Arabic HTML segment missing dir='rtl' on <html> tag.",
            }
    return None


def run_deterministic_qa(
    translated_results: dict[str, list[dict[str, Any]]],
    protected_terms: list[str],
) -> list[dict[str, Any]]:
    """
    Run all deterministic checks.
    Returns a flat list of QA issue records.
    """
    issues: list[dict[str, Any]] = []
    critical_count = 0

    for lang_code, segments in translated_results.items():
        for seg in segments:
            seg_id = seg["segment_id"]
            src = seg.get("source_text", "")
            trn = seg.get("translated_text", "")
            base = {
                "segment_id": seg_id,
                "language_code": lang_code,
                "source_text": src[:200],
                "translated_text": trn[:200],
            }

            checks = [
                _check_empty(seg),
                _check_untranslated(seg),
                _check_broken_html(seg),
                _check_rtl(seg, lang_code),
            ]
            for result in checks:
                if result:
                    record = {**base, **result}
                    issues.append(record)
                    if result["severity"] == "critical":
                        critical_count += 1

            for result in _check_placeholder_corruption(seg):
                issues.append({**base, **result})
                if result["severity"] == "critical":
                    critical_count += 1

            for result in _check_url_preservation(seg):
                issues.append({**base, **result})
                if result["severity"] == "critical":
                    critical_count += 1

            # Check protected terms are preserved
            for term in protected_terms:
                if term in src and term not in trn:
                    record = {
                        **base,
                        "issue_type": "protected_term_altered",
                        "severity": "critical",
                        "details": f"Protected term '{term}' present in source but absent/altered in translation.",
                    }
                    issues.append(record)
                    critical_count += 1

    if critical_count > 0:
        print(f"\n  *** {critical_count} CRITICAL QA ISSUES FOUND ***")
        for iss in issues:
            if iss["severity"] == "critical":
                print(f"    [CRITICAL] {iss['language_code']} | {iss['segment_id'][:8]}… | {iss['issue_type']}: {iss['details'][:100]}")

    return issues


def save_qa_report(
    issues: list[dict[str, Any]],
    translated_results: dict[str, list[dict[str, Any]]],
    path: Path = QA_REPORT_JSON,
) -> None:
    # Count totals
    total_segs = sum(len(v) for v in translated_results.values())
    critical = sum(1 for i in issues if i["severity"] == "critical")
    warning = sum(1 for i in issues if i["severity"] == "warning")
    info = sum(1 for i in issues if i["severity"] == "info")

    report = {
        "generated_at": _now_iso(),
        "summary": {
            "total_segments_checked": total_segs,
            "total_issues": len(issues),
            "critical": critical,
            "warning": warning,
            "info": info,
        },
        "issues": issues,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  QA report saved -> {path}  ({len(issues)} issues)")


# ---------------------------------------------------------------------------
# LLM-assisted QA (stretch goal)
# ---------------------------------------------------------------------------


def run_llm_qa(
    translated_results: dict[str, list[dict[str, Any]]],
    languages: list[dict[str, Any]],
    sample_size: int = 5,
) -> list[dict[str, Any]]:
    """
    Run LLM-assisted QA on a sample of translated segments.
    Checks grammar, fluency, tone, and meaning preservation.
    """
    try:
        import os
        from google import genai as _genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("  [SKIP] LLM QA: GEMINI_API_KEY not set.")
            return []
        qa_client = _genai.Client(api_key=api_key)
    except ImportError:
        print("  [SKIP] LLM QA: google-genai not installed.")
        return []

    lang_map = {lang["code"]: lang["name"] for lang in languages}
    llm_issues: list[dict[str, Any]] = []

    for lang_code, segments in translated_results.items():
        lang_name = lang_map.get(lang_code, lang_code)
        sample = [s for s in segments if s.get("translated_text", "").strip()][:sample_size]

        print(f"  LLM QA: {lang_name} ({len(sample)} samples)…")

        for seg in sample:
            prompt = (
                f"You are a professional translator and QA reviewer.\n"
                f"Review this English-to-{lang_name} translation for a financial trading website.\n\n"
                f"Source (English):\n{seg['source_text'][:500]}\n\n"
                f"Translation ({lang_name}):\n{seg['translated_text'][:500]}\n\n"
                "Check for: grammar errors, fluency issues, tone mismatch, cultural inappropriateness, "
                "meaning changes, or missing content.\n\n"
                "Respond ONLY with a JSON object (no markdown fences) in this exact format:\n"
                '{"pass": true/false, "issues": [{"type": "string", "severity": "critical|warning|info", '
                '"detail": "string", "suggestion": "string"}]}'
            )
            prompt_hash = _sha256(prompt)
            call_record: dict[str, Any] = {
                "stage": "LLM_QA",
                "page_url": seg.get("page_url"),
                "language_code": lang_code,
                "segment_id": seg["segment_id"],
                "timestamp": _now_iso(),
                "provider": "google",
                "model": GEMINI_MODEL,
                "prompt_hash": prompt_hash,
                "input_artifacts": [str(ROOT / "translations" / lang_code / "translated_segments.json")],
                "output_artifact": str(LLM_QA_REPORT_JSON),
                "estimated_input_tokens": _estimate_tokens(prompt),
                "estimated_output_tokens": 300,
            }

            try:
                response = qa_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                result = json.loads(raw)

                for iss in result.get("issues", []):
                    llm_issues.append(
                        {
                            "segment_id": seg["segment_id"],
                            "language_code": lang_code,
                            "source_text": seg["source_text"][:200],
                            "translated_text": seg["translated_text"][:200],
                            "severity": iss.get("severity", "info"),
                            "issue_type": f"llm_qa_{iss.get('type', 'unknown')}",
                            "details": iss.get("detail", ""),
                            "suggestion": iss.get("suggestion", ""),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"    [WARN] LLM QA call failed for {seg['segment_id'][:8]}: {exc}")

            _append_llm_call(call_record)

    return llm_issues


def save_llm_qa_report(issues: list[dict[str, Any]], path: Path = LLM_QA_REPORT_JSON) -> None:
    report = {
        "generated_at": _now_iso(),
        "total_issues": len(issues),
        "issues": issues,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  LLM QA report saved -> {path}  ({len(issues)} issues)")