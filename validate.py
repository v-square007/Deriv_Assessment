"""
validate.py
-----------
Validation script for the Deriv multilingual translation pipeline.

Checks:
  - Required artifacts exist
  - JSON files are valid and conform to expected schemas
  - At least 1-2 pages were processed
  - Arabic is present when <=1 language configured
  - Source metadata preserved in segments
  - Protected terms unchanged in all translations
  - Placeholders not corrupted
  - URLs preserved
  - Translated output files exist
  - Arabic output includes RTL handling (dir="rtl" or dir='rtl')
  - QA report generated
  - Critical QA issues surfaced
  - LLM call logs exist
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = 0
failed = 0
warnings = 0


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str) -> None:
    global failed
    failed += 1
    print(f"  {RED}✗{RESET}  {msg}")


def warn(msg: str) -> None:
    global warnings
    warnings += 1
    print(f"  {YELLOW}!{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 60)


def load_json(path: Path) -> Any | None:
    if not path.exists():
        fail(f"Missing: {path.name}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ok(f"Valid JSON: {path.name}")
        return data
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path.name}: {exc}")
        return None


def check_file_exists(path: Path, label: str | None = None) -> bool:
    label = label or path.name
    if path.exists():
        ok(f"Exists: {label}")
        return True
    fail(f"Missing: {label}")
    return False


def check_required_artifacts() -> None:
    section("1. Required artifacts")
    for p in [
        ROOT / "pages.json",
        ROOT / "target_languages.json",
        ROOT / "extracted_segments.json",
        ROOT / "protected_terms.json",
        ROOT / "translations",
        ROOT / "output",
        ROOT / "qa_report.json",
        ROOT / "llm_calls.jsonl",
    ]:
        check_file_exists(p)


def check_config_files() -> tuple[list[str], list[dict[str, Any]]]:
    section("2. Config files")
    pages: list[str] = []
    languages: list[dict[str, Any]] = []
    pages_data = load_json(ROOT / "pages.json")
    if pages_data:
        if "pages" in pages_data and isinstance(pages_data["pages"], list):
            pages = pages_data["pages"]
            ok(f"pages.json has {len(pages)} pages")
        else:
            fail("pages.json missing 'pages' array")
    lang_data = load_json(ROOT / "target_languages.json")
    if lang_data:
        if "target_languages" in lang_data and isinstance(lang_data["target_languages"], list):
            languages = lang_data["target_languages"]
            ok(f"target_languages.json has {len(languages)} languages")
        else:
            fail("target_languages.json missing 'target_languages' array")
    return pages, languages


def check_segments(pages: list[str]) -> list[dict[str, Any]]:
    section("3. Extracted segments")
    segments: list[dict[str, Any]] = []
    data = load_json(ROOT / "extracted_segments.json")
    if data is None:
        return segments

    if not isinstance(data, list):
        fail("extracted_segments.json should be a JSON array")
        return segments

    segments = data
    ok(f"extracted_segments.json has {len(segments)} segments")

    if len(segments) == 0:
        fail("No extracted segments found")
        return segments

    processed_pages = set(s.get("page_url") for s in segments if s.get("page_url"))
    if len(processed_pages) >= 1:
        ok(f"{len(processed_pages)} page(s) processed")
    else:
        fail("No pages appear to have been processed (no page_url in segments)")

    if pages and len(processed_pages) < min(2, len(pages)):
        warn(
            f"Only {len(processed_pages)} page(s) produced segments out of {len(pages)} configured"
        )

    required_fields = {
        "segment_id",
        "page_url",
        "html_path",
        "source_text",
        "contains_html",
        "placeholders",
        "links",
    }
    schema_errors = 0
    for i, seg in enumerate(segments[:25]):
        missing = required_fields - set(seg.keys())
        if missing:
            schema_errors += 1
            if schema_errors <= 3:
                fail(f"Segment[{i}] missing fields: {sorted(missing)}")
    if schema_errors == 0:
        ok("Segment schema valid (spot-checked first 25)")
    elif schema_errors > 3:
        fail(f"...and {schema_errors - 3} more segment schema errors")

    has_source = all((s.get("source_text") or "").strip() for s in segments)
    if has_source:
        ok("All segments have source_text")
    else:
        fail("Some segments are missing source_text")
    return segments


def check_protected_terms() -> list[str]:
    section("4. Protected terms")
    data = load_json(ROOT / "protected_terms.json")
    if not data or "protected_terms" not in data:
        fail("protected_terms.json missing 'protected_terms' key")
        return []
    terms = data["protected_terms"]
    if "Deriv" in terms:
        ok("'Deriv' is in protected terms")
    else:
        fail("'Deriv' not found in protected_terms")
    return terms


def check_translations(languages: list[dict[str, Any]], protected_terms: list[str], segments: list[dict[str, Any]]) -> None:
    _ = segments
    section("5. Translations")
    lang_codes = [l["code"] for l in languages]
    if len(lang_codes) <= 1 and "ar" not in lang_codes:
        fail("Arabic must be included when only one language is selected")
    elif "ar" in lang_codes:
        ok("Arabic is included in target languages")

    url_re = re.compile(r'https?://[^\s"\'<>]+')
    ph_re = re.compile(r"\[\[PROTECTED_\d+\]\]")

    for lang in languages:
        code = lang["code"]
        seg_file = ROOT / "translations" / code / "translated_segments.json"
        if not seg_file.exists():
            fail(f"translations/{code}/translated_segments.json not found")
            continue
        data = load_json(seg_file)
        if isinstance(data, list):
            ok(f"translations/{code}/: {len(data)} segments")

            req = {"segment_id", "language_code", "source_text", "translated_text"}
            schema_errors = 0
            for i, seg in enumerate(data[:10]):
                missing = req - set(seg.keys())
                if missing:
                    schema_errors += 1
                    if schema_errors <= 3:
                        fail(f"[{code}] Segment[{i}] missing fields: {sorted(missing)}")
            if schema_errors == 0:
                ok(f"[{code}] Translated segment schema valid")

            term_issues = 0
            for seg in data:
                src = seg.get("source_text", "")
                trn = seg.get("translated_text", "")
                for term in protected_terms:
                    if term in src and term not in trn:
                        term_issues += 1
                        if term_issues <= 2:
                            warn(f"[{code}] Protected term '{term}' altered in {seg.get('segment_id', '')[:8]}")
            if term_issues == 0:
                ok(f"[{code}] Protected terms unchanged")
            else:
                fail(f"[{code}] {term_issues} protected term violations")

            placeholder_issues = sum(1 for seg in data if ph_re.search(seg.get("translated_text", "")))
            if placeholder_issues == 0:
                ok(f"[{code}] No unreplaced placeholders")
            else:
                fail(f"[{code}] {placeholder_issues} segments still contain [[PROTECTED_N]] placeholders")

            url_issues = 0
            for seg in data:
                src_urls = set(url_re.findall(seg.get("source_text", "")))
                trn_urls = set(url_re.findall(seg.get("translated_text", "")))
                url_issues += len(src_urls - trn_urls)
            if url_issues == 0:
                ok(f"[{code}] URLs preserved")
            else:
                warn(f"[{code}] {url_issues} URLs may be missing from translations")


def check_html_output(languages: list[dict[str, Any]]) -> None:
    section("6. HTML output & RTL")
    for lang in languages:
        code = lang["code"]
        is_rtl = lang.get("direction", "ltr") == "rtl"
        lang_dir = ROOT / "output" / code
        if not lang_dir.exists():
            fail(f"output/{code}/ directory missing")
            continue
        html_files = list(lang_dir.glob("*.html"))
        if not html_files:
            fail(f"output/{code}/: No HTML files found")
            continue
        ok(f"output/{code}/: {len(html_files)} HTML file(s)")
        if is_rtl:
            rtl_ok_count = 0
            for hf in html_files:
                content = hf.read_text(encoding="utf-8")
                if 'dir="rtl"' in content or "dir='rtl'" in content:
                    rtl_ok_count += 1
            if rtl_ok_count == len(html_files):
                ok(f"[{code}] All HTML files have dir='rtl'")
            else:
                fail(f"[{code}] Arabic HTML output missing dir='rtl' attribute")


def check_qa_report() -> None:
    section("7. QA report")
    data = load_json(ROOT / "qa_report.json")
    if not data:
        return
    if "issues" not in data:
        fail("qa_report.json missing 'issues' key")
        return

    issues = data["issues"]
    ok(f"qa_report.json has {len(issues)} issue(s)")
    summary = data.get("summary", {})
    if summary:
        ok(
            f"QA summary: {summary.get('critical', '?')} critical, "
            f"{summary.get('warning', '?')} warning"
        )

    critical = [i for i in issues if i.get("severity") == "critical"]
    if critical:
        warn(f"{len(critical)} critical QA issues present — should be visible in terminal output")
        for iss in critical[:3]:
            warn(
                f"  [{iss.get('language_code')}] {iss.get('issue_type')}: "
                f"{str(iss.get('details', ''))[:80]}"
            )
    else:
        ok("No critical QA issues")


def check_llm_logs() -> None:
    section("8. LLM call logs")
    llm_path = ROOT / "llm_calls.jsonl"
    if not llm_path.exists():
        fail("llm_calls.jsonl not found")
        return
    lines = [l for l in llm_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        fail("llm_calls.jsonl is empty")
        return
    records: list[dict[str, Any]] = []
    parse_errors = 0
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1

    if parse_errors == 0:
        ok(f"llm_calls.jsonl: {len(records)} valid records")
    else:
        fail(f"llm_calls.jsonl: {parse_errors} parse errors in {len(lines)} lines")

    req_fields = {
        "stage",
        "timestamp",
        "provider",
        "model",
        "prompt_hash",
        "estimated_input_tokens",
        "estimated_output_tokens",
    }
    field_errors = 0
    for r in records:
        missing = req_fields - set(r.keys())
        if missing:
            field_errors += 1
    if field_errors == 0:
        ok("All LLM call records have required fields")
    else:
        fail(f"{field_errors}/{len(records)} LLM records missing required fields")

    stages_logged = set(r.get("stage") for r in records)
    if "TRANSLATION" in stages_logged:
        ok(f"Translation calls logged. Stages: {stages_logged}")
    else:
        warn(f"No TRANSLATION stage found in logs. Stages present: {stages_logged}")


def check_optional_artifacts() -> None:
    section("9. Optional artifacts")
    for p in [ROOT / "cost_report.json", ROOT / "translation_cache.json", ROOT / "llm_qa_report.json", ROOT / "run_metrics.json"]:
        if p.exists():
            ok(f"Present: {p.name}")
        else:
            warn(f"Optional artifact absent: {p.name}")


def main() -> int:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Deriv Translation Pipeline – Validation{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    check_required_artifacts()
    pages, languages = check_config_files()
    segments = check_segments(pages)
    protected_terms = check_protected_terms()
    check_translations(languages, protected_terms, segments)
    check_html_output(languages)
    check_qa_report()
    check_llm_logs()
    check_optional_artifacts()
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {GREEN}Passed  : {passed}{RESET}")
    print(f"  {YELLOW}Warnings: {warnings}{RESET}")
    print(f"  {RED}Failed  : {failed}{RESET}")
    if failed > 0:
        print(f"\n  {RED}{BOLD}VALIDATION FAILED — {failed} check(s) did not pass.{RESET}")
        return 1
    if warnings > 0:
        print(f"\n  {YELLOW}{BOLD}VALIDATION PASSED WITH WARNINGS{RESET}")
        return 0
    print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
