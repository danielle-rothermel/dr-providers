import importlib.util
import sys
from pathlib import Path
from typing import cast

CORPUS_DIR = Path("data/audit-corpus")
GROUND_TRUTH_DIR = CORPUS_DIR / "ground-truth"
SCRIPT_PATH = Path("scripts/generate_audit_ground_truth.py")

SPEC = importlib.util.spec_from_file_location(
    "generate_audit_ground_truth", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
audit_ground_truth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_ground_truth
SPEC.loader.exec_module(audit_ground_truth)

CanonicalSuggestionSet = audit_ground_truth.CanonicalSuggestionSet
Analysis = audit_ground_truth.Analysis
ParsedAuditSet = audit_ground_truth.ParsedAuditSet
analyze = audit_ground_truth.analyze
parse_corpus = audit_ground_truth.parse_corpus


def test_parse_corpus_extracts_all_audit_metadata() -> None:
    parsed = parse_corpus(CORPUS_DIR)

    assert parsed.corpus_id == "dr-providers-ponytail-audit-gpt-5.5"
    assert len(parsed.audits) == 18
    assert sum(audit.suggestion_count for audit in parsed.audits) == 167

    first_audit = parsed.audits[0]
    assert first_audit.audit_id == "gpt-5.5/off/audit_0"
    assert first_audit.stats.model == "gpt-5.5"
    assert first_audit.stats.thinking_level == "off"
    assert first_audit.stats.cost_usd == 0.53
    assert first_audit.stats.throughput_tok_s == 445
    assert first_audit.net_estimate.lines == -220
    assert first_audit.suggestion_kind_counts == {
        "delete": 6,
        "shrink": 1,
        "stdlib": 1,
        "yagni": 2,
    }


def test_canonical_suggestions_cover_every_parsed_suggestion() -> None:
    parsed = parse_corpus(CORPUS_DIR)
    canonical = CanonicalSuggestionSet.model_validate_json(
        (GROUND_TRUTH_DIR / "canonical_suggestions.json").read_text()
    )

    analysis = analyze(parsed, canonical)

    assert analysis.parsed_suggestion_count == 167
    assert analysis.canonical_suggestion_count == 21


def test_generated_json_artifacts_match_committed_outputs() -> None:
    parsed = parse_corpus(CORPUS_DIR)
    canonical = CanonicalSuggestionSet.model_validate_json(
        (GROUND_TRUTH_DIR / "canonical_suggestions.json").read_text()
    )
    analysis = analyze(parsed, canonical)

    committed_parsed = ParsedAuditSet.model_validate_json(
        (GROUND_TRUTH_DIR / "parsed_audits.json").read_text()
    )
    committed_analysis = Analysis.model_validate_json(
        (GROUND_TRUTH_DIR / "analysis.json").read_text()
    )

    assert parsed == committed_parsed
    assert analysis == committed_analysis


def test_analysis_answers_first_pass_benchmark_questions() -> None:
    parsed = parse_corpus(CORPUS_DIR)
    canonical = CanonicalSuggestionSet.model_validate_json(
        (GROUND_TRUTH_DIR / "canonical_suggestions.json").read_text()
    )
    analysis = analyze(parsed, canonical)

    answers = {
        question.question_id: question for question in analysis.questions
    }

    assert answers["cost_off_vs_high"].values["savings_percent"] == 26.5
    assert answers["speed_low_vs_minimal"].values["percent_delta"] == -5.2
    minimal_coverage = cast(
        "dict[str, int]",
        answers["coverage_at_k_by_level"].values["minimal"],
    )
    xhigh_coverage = cast(
        "dict[str, int]",
        answers["coverage_at_k_by_level"].values["xhigh"],
    )

    assert minimal_coverage["k=3"] == 10
    assert xhigh_coverage["k=3"] == 14
