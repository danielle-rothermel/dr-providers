import importlib.util
import shutil
import sys
from pathlib import Path
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "audit-corpus"
GROUND_TRUTH_DIR = CORPUS_DIR / "ground-truth"
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_audit_ground_truth.py"

SPEC = importlib.util.spec_from_file_location(
    "generate_audit_ground_truth", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
audit_ground_truth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_ground_truth
SPEC.loader.exec_module(audit_ground_truth)

CanonicalSuggestionSet = audit_ground_truth.CanonicalSuggestionSet
ParsedAuditSet = audit_ground_truth.ParsedAuditSet
analyze = audit_ground_truth.analyze
parse_corpus = audit_ground_truth.parse_corpus

SYNTHETIC_SUGGESTION_ID = "audit/0/suggestion_0"


def _synthetic_parsed() -> ParsedAuditSet:
    return ParsedAuditSet.model_validate(
        {
            "corpus_id": "synthetic",
            "generated_from_manifest": "synthetic/manifest.json",
            "audits": [
                {
                    "audit_id": "audit/0",
                    "source_path": "audit.md",
                    "thinking_level": "off",
                    "stats": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "reasoning_tokens": 0,
                        "cache_hit_percent": 0.0,
                        "cost_usd": 0.0,
                        "cost_mode": "synthetic",
                        "context_percent": 0.0,
                        "context_limit_tokens": 1,
                        "context_mode": "synthetic",
                        "agent": "synthetic",
                        "model": "synthetic",
                        "thinking_level": "off",
                        "status": "Done",
                        "throughput_tok_s": 1.0,
                    },
                    "net_estimate": {
                        "approximate": False,
                        "lines": 0,
                        "dependencies": 0,
                    },
                    "suggestion_count": 1,
                    "suggestion_kind_counts": {"delete": 1},
                    "suggestions": [
                        {
                            "suggestion_id": SYNTHETIC_SUGGESTION_ID,
                            "audit_id": "audit/0",
                            "index": 0,
                            "kind": "delete",
                            "text": "synthetic suggestion",
                            "rationale": None,
                            "action": None,
                            "replacement": None,
                            "files": [],
                            "raw": "synthetic suggestion",
                        }
                    ],
                    "raw_header": "synthetic header",
                    "raw_done_line": "Done",
                    "raw_net_line": "net: +0 lines, +0 deps possible.",
                }
            ],
        }
    )


def _synthetic_canonical(
    *source_groups: tuple[str, ...],
    canonical_ids: tuple[str, ...] | None = None,
) -> CanonicalSuggestionSet:
    ids = canonical_ids or tuple(
        f"canonical_{index}" for index in range(len(source_groups))
    )
    return CanonicalSuggestionSet.model_validate(
        {
            "schema_version": 1,
            "corpus_id": "synthetic",
            "canonical_suggestions": [
                {
                    "canonical_id": canonical_id,
                    "canonical_kind": "delete",
                    "summary": "synthetic",
                    "files": [],
                    "resolution": "fully_addressed",
                    "resolution_notes": "synthetic",
                    "source_suggestions": [
                        {
                            "suggestion_id": source_id,
                            "audit_id": "audit/0",
                            "index": index,
                        }
                        for index, source_id in enumerate(source_ids)
                    ],
                    "scope_notes": None,
                }
                for canonical_id, source_ids in zip(
                    ids, source_groups, strict=True
                )
            ],
        }
    )


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


def test_generated_artifacts_exactly_match_committed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copyfile(
        GROUND_TRUTH_DIR / "canonical_suggestions.json",
        tmp_path / "canonical_suggestions.json",
    )

    # Recorded source paths require the canonical repository-relative argument.
    monkeypatch.chdir(REPO_ROOT)
    audit_ground_truth.main(
        corpus_dir=Path("data/audit-corpus"), output_dir=tmp_path
    )

    for filename in ("parsed_audits.json", "analysis.json", "analysis.md"):
        assert (tmp_path / filename).read_bytes() == (
            GROUND_TRUTH_DIR / filename
        ).read_bytes()


@pytest.mark.parametrize(
    ("canonical", "message"),
    [
        (
            _synthetic_canonical(),
            "Canonical mapping misses suggestions",
        ),
        (
            _synthetic_canonical(
                (SYNTHETIC_SUGGESTION_ID, "audit/0/suggestion_unknown")
            ),
            "Canonical mapping has unknown suggestions",
        ),
        (
            _synthetic_canonical(
                (SYNTHETIC_SUGGESTION_ID,),
                (SYNTHETIC_SUGGESTION_ID,),
                canonical_ids=("duplicate", "duplicate"),
            ),
            "Duplicate canonical ids",
        ),
    ],
    ids=("missing-source", "unknown-source", "duplicate-canonical-id"),
)
def test_canonical_mapping_integrity_rejects_invalid_inputs(
    canonical: CanonicalSuggestionSet,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze(_synthetic_parsed(), canonical)


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
