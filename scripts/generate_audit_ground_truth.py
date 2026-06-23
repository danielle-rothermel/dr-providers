from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict

AUDIT_HEADER_PATTERN = re.compile(
    r"^↑(?P<input>\d+(?:\.\d+)?)k "
    r"↓(?P<output>\d+(?:\.\d+)?)k "
    r"R(?P<reasoning>\d+(?:\.\d+)?)k "
    r"CH(?P<cache_hit>\d+(?:\.\d+)?)% "
    r"\$(?P<cost>\d+(?:\.\d+)?) "
    r"\((?P<cost_mode>[^)]+)\) "
    r"(?P<context_percent>\d+(?:\.\d+)?)%/"
    r"(?P<context_limit>\d+(?:\.\d+)?)k "
    r"\((?P<context_mode>[^)]+)\)\s+"
    r"\((?P<agent>[^)]+)\) "
    r"(?P<model>\S+) • (?P<thinking>.+)$"
)
DONE_PATTERN = re.compile(
    r"^(?P<status>\w+) — (?P<throughput>\d+(?:\.\d+)?) tok/s$"
)
NET_PATTERN = re.compile(
    r"^net: (?P<approx>~)?(?P<lines>[+-]?\d+) lines, "
    r"(?P<deps>[+-]?\d+) (?:direct )?deps? possible\.$"
)
SUGGESTION_PATTERN = re.compile(r"^`?(?P<kind>[a-z_]+):?`?\s+(?P<body>.+)$")
FILES_PATTERN = re.compile(r"\s+`?\[(?P<files>[^\[\]]+)\]`?$")
FIRST_SENTENCE_PARTS = 2
MIN_AUDIT_LINES = 5


class Resolution(StrEnum):
    FULLY_ADDRESSED = "fully_addressed"
    PARTIALLY_OR_DIFFERENTLY_ADDRESSED = "partially_or_differently_addressed"
    NOT_ADDRESSED = "not_addressed"


class AuditManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    level: str
    path: str


class AuditManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    description: str
    source_repo: str
    agent: str
    model: str
    prompt: str
    ponytail_level: str
    levels: list[str]
    runs_per_level: int
    files: list[AuditManifestFile]


class RunStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_hit_percent: float
    cost_usd: float
    cost_mode: str
    context_percent: float
    context_limit_tokens: int
    context_mode: str
    agent: str
    model: str
    thinking_level: str
    status: str
    throughput_tok_s: float


class NetEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approximate: bool
    lines: int
    dependencies: int


class ParsedSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestion_id: str
    audit_id: str
    index: int
    kind: str
    text: str
    rationale: str | None
    action: str | None
    replacement: str | None
    files: list[str]
    raw: str


class ParsedAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    source_path: str
    thinking_level: str
    stats: RunStats
    net_estimate: NetEstimate
    suggestion_count: int
    suggestion_kind_counts: dict[str, int]
    suggestions: list[ParsedSuggestion]
    raw_header: str
    raw_done_line: str
    raw_net_line: str


class ParsedAuditSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    generated_from_manifest: str
    audits: list[ParsedAudit]


class SourceSuggestionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestion_id: str
    audit_id: str
    index: int


class CanonicalSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    canonical_kind: str
    summary: str
    files: list[str]
    resolution: Resolution
    resolution_notes: str
    source_suggestions: list[SourceSuggestionRef]
    scope_notes: str | None = None


class CanonicalSuggestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    corpus_id: str
    canonical_suggestions: list[CanonicalSuggestion]


class QuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    method: str
    answer: str
    values: dict[str, object]


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    audit_count: int
    parsed_suggestion_count: int
    canonical_suggestion_count: int
    questions: list[QuestionAnswer]


def _tokens_from_k(raw_value: str) -> int:
    return round(float(raw_value) * 1000)


def _load_manifest(corpus_dir: Path) -> AuditManifest:
    manifest_path = corpus_dir / "manifest.json"
    return AuditManifest.model_validate_json(manifest_path.read_text())


def _parse_header(header: str, done_line: str) -> RunStats:
    header_match = AUDIT_HEADER_PATTERN.match(header)
    if header_match is None:
        raise ValueError(f"Could not parse audit header: {header}")
    done_match = DONE_PATTERN.match(done_line)
    if done_match is None:
        raise ValueError(f"Could not parse done line: {done_line}")

    thinking_level = header_match["thinking"]
    if thinking_level.startswith("thinking "):
        thinking_level = thinking_level.removeprefix("thinking ")

    return RunStats(
        input_tokens=_tokens_from_k(header_match["input"]),
        output_tokens=_tokens_from_k(header_match["output"]),
        reasoning_tokens=_tokens_from_k(header_match["reasoning"]),
        cache_hit_percent=float(header_match["cache_hit"]),
        cost_usd=float(header_match["cost"]),
        cost_mode=header_match["cost_mode"],
        context_percent=float(header_match["context_percent"]),
        context_limit_tokens=_tokens_from_k(header_match["context_limit"]),
        context_mode=header_match["context_mode"],
        agent=header_match["agent"],
        model=header_match["model"],
        thinking_level=thinking_level,
        status=done_match["status"],
        throughput_tok_s=float(done_match["throughput"]),
    )


def _parse_net_estimate(raw_net_line: str) -> NetEstimate:
    net_match = NET_PATTERN.match(raw_net_line)
    if net_match is None:
        raise ValueError(f"Could not parse net line: {raw_net_line}")
    return NetEstimate(
        approximate=net_match["approx"] is not None,
        lines=int(net_match["lines"]),
        dependencies=int(net_match["deps"]),
    )


def _parse_files(raw_body: str) -> tuple[str, list[str]]:
    files_match = FILES_PATTERN.search(raw_body)
    if files_match is None:
        return raw_body.strip(), []
    files = [
        item.strip().strip("`")
        for item in files_match["files"].split(",")
        if item.strip()
    ]
    return raw_body[: files_match.start()].strip(), files


def _split_suggestion_text(
    text: str,
) -> tuple[str | None, str | None, str | None]:
    replacement = None
    main_text = text
    if "Replacement:" in text:
        main_text, replacement = text.split("Replacement:", maxsplit=1)
        main_text = main_text.strip()
        replacement = replacement.strip()

    first_sentence = main_text.split(".", maxsplit=1)
    if len(first_sentence) == FIRST_SENTENCE_PARTS:
        rationale = first_sentence[0].strip()
        action = first_sentence[1].strip()
    else:
        rationale = None
        action = main_text.strip() or None
    return rationale or None, action or None, replacement or None


def _parse_suggestion(
    *, audit_id: str, index: int, raw_suggestion: str
) -> ParsedSuggestion:
    suggestion_match = SUGGESTION_PATTERN.match(raw_suggestion)
    if suggestion_match is None:
        raise ValueError(f"Could not parse suggestion: {raw_suggestion}")
    text, files = _parse_files(suggestion_match["body"])
    rationale, action, replacement = _split_suggestion_text(text)
    return ParsedSuggestion(
        suggestion_id=f"{audit_id}#s{index:02d}",
        audit_id=audit_id,
        index=index,
        kind=suggestion_match["kind"],
        text=text,
        rationale=rationale,
        action=action,
        replacement=replacement,
        files=files,
        raw=raw_suggestion,
    )


def _parse_audit(
    corpus_dir: Path, manifest_file: AuditManifestFile
) -> ParsedAudit:
    source_path = corpus_dir / manifest_file.path
    lines = source_path.read_text().splitlines()
    if len(lines) < MIN_AUDIT_LINES:
        raise ValueError(f"Audit file is too short: {source_path}")

    raw_header = lines[1]
    raw_done_line = lines[2]
    raw_net_line = next(
        line for line in reversed(lines) if line.startswith("net:")
    )
    suggestion_lines = [
        line
        for line in lines[4:]
        if line.strip() and not line.startswith("net:")
    ]
    suggestions = [
        _parse_suggestion(
            audit_id=manifest_file.audit_id,
            index=index,
            raw_suggestion=raw_suggestion,
        )
        for index, raw_suggestion in enumerate(suggestion_lines, start=1)
    ]
    kind_counts = Counter(suggestion.kind for suggestion in suggestions)

    return ParsedAudit(
        audit_id=manifest_file.audit_id,
        source_path=str(source_path),
        thinking_level=manifest_file.level,
        stats=_parse_header(raw_header, raw_done_line),
        net_estimate=_parse_net_estimate(raw_net_line),
        suggestion_count=len(suggestions),
        suggestion_kind_counts=dict(sorted(kind_counts.items())),
        suggestions=suggestions,
        raw_header=raw_header,
        raw_done_line=raw_done_line,
        raw_net_line=raw_net_line,
    )


def parse_corpus(corpus_dir: Path) -> ParsedAuditSet:
    manifest = _load_manifest(corpus_dir)
    return ParsedAuditSet(
        corpus_id=manifest.corpus_id,
        generated_from_manifest=str(corpus_dir / "manifest.json"),
        audits=[
            _parse_audit(corpus_dir, manifest_file)
            for manifest_file in manifest.files
        ],
    )


def _load_canonical_suggestions(
    output_dir: Path,
) -> CanonicalSuggestionSet:
    canonical_path = output_dir / "canonical_suggestions.json"
    return CanonicalSuggestionSet.model_validate_json(
        canonical_path.read_text()
    )


def _validate_canonical_suggestions(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> None:
    parsed_ids = {
        suggestion.suggestion_id
        for audit in parsed.audits
        for suggestion in audit.suggestions
    }
    mapped_ids = {
        source.suggestion_id
        for suggestion in canonical.canonical_suggestions
        for source in suggestion.source_suggestions
    }
    missing = sorted(parsed_ids - mapped_ids)
    unknown = sorted(mapped_ids - parsed_ids)
    if missing:
        raise ValueError(f"Canonical mapping misses suggestions: {missing}")
    if unknown:
        raise ValueError(
            f"Canonical mapping has unknown suggestions: {unknown}"
        )

    canonical_ids = [
        suggestion.canonical_id
        for suggestion in canonical.canonical_suggestions
    ]
    duplicate_ids = [
        item for item, count in Counter(canonical_ids).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate canonical ids: {sorted(duplicate_ids)}")


def _level_groups(parsed: ParsedAuditSet) -> dict[str, list[ParsedAudit]]:
    levels: dict[str, list[ParsedAudit]] = defaultdict(list)
    for audit in parsed.audits:
        levels[audit.thinking_level].append(audit)
    return dict(sorted(levels.items()))


def _canonical_lookup(
    canonical: CanonicalSuggestionSet,
) -> dict[str, list[CanonicalSuggestion]]:
    by_source: dict[str, list[CanonicalSuggestion]] = defaultdict(list)
    for suggestion in canonical.canonical_suggestions:
        for source in suggestion.source_suggestions:
            by_source[source.suggestion_id].append(suggestion)
    return by_source


def _canonical_ids_for_audit(
    audit: ParsedAudit,
    by_source: dict[str, list[CanonicalSuggestion]],
    *,
    addressed_only: bool = False,
) -> set[str]:
    ids: set[str] = set()
    for suggestion in audit.suggestions:
        for canonical in by_source[suggestion.suggestion_id]:
            if (
                addressed_only
                and canonical.resolution == Resolution.NOT_ADDRESSED
            ):
                continue
            ids.add(canonical.canonical_id)
    return ids


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3)


def _cost_comparison(parsed: ParsedAuditSet) -> QuestionAnswer:
    levels = _level_groups(parsed)
    off_cost = _mean([audit.stats.cost_usd for audit in levels["off"]])
    high_cost = _mean([audit.stats.cost_usd for audit in levels["high"]])
    savings_percent = round(((high_cost - off_cost) / high_cost) * 100, 1)
    return QuestionAnswer(
        question_id="cost_off_vs_high",
        question="How much cheaper is thinking off than high thinking?",
        method="Compare mean run cost across the three off and high audits.",
        answer=(
            f"Thinking off averages ${off_cost:.3f}, high averages "
            f"${high_cost:.3f}; off is {savings_percent}% cheaper."
        ),
        values={
            "off_mean_cost_usd": off_cost,
            "high_mean_cost_usd": high_cost,
            "savings_percent": savings_percent,
        },
    )


def _speed_comparison(parsed: ParsedAuditSet) -> QuestionAnswer:
    levels = _level_groups(parsed)
    low_speed = _mean(
        [audit.stats.throughput_tok_s for audit in levels["low"]]
    )
    minimal_speed = _mean(
        [audit.stats.throughput_tok_s for audit in levels["minimal"]]
    )
    speedup = round(low_speed / minimal_speed, 3)
    percent_delta = round(
        ((low_speed - minimal_speed) / minimal_speed) * 100, 1
    )
    if percent_delta >= 0:
        comparison = f"low is {percent_delta}% faster"
    else:
        comparison = f"low is {abs(percent_delta)}% slower"
    return QuestionAnswer(
        question_id="speed_low_vs_minimal",
        question="How much faster is low thinking than minimal thinking?",
        method=(
            "Compare mean throughput in tokens per second across the three "
            "low and minimal audits."
        ),
        answer=(
            f"Low averages {low_speed:.1f} tok/s, minimal averages "
            f"{minimal_speed:.1f} tok/s; {comparison}."
        ),
        values={
            "low_mean_tok_s": low_speed,
            "minimal_mean_tok_s": minimal_speed,
            "speedup_ratio": speedup,
            "percent_delta": percent_delta,
        },
    )


def _raw_suggestion_counts(parsed: ParsedAuditSet) -> QuestionAnswer:
    values: dict[str, object] = {
        audit.audit_id: audit.suggestion_count for audit in parsed.audits
    }
    return QuestionAnswer(
        question_id="raw_suggestion_counts",
        question="How many raw suggestions did each run produce?",
        method="Count parsed suggestion lines in each audit.",
        answer=(
            f"Runs produced between {min(values.values())} and "
            f"{max(values.values())} raw suggestions."
        ),
        values=values,
    )


def _canonical_coverage(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> QuestionAnswer:
    by_source = _canonical_lookup(canonical)
    coverage_values = {
        audit.audit_id: len(_canonical_ids_for_audit(audit, by_source))
        for audit in parsed.audits
    }
    values: dict[str, object] = dict(coverage_values)
    max_audit_id = max(coverage_values, key=coverage_values.__getitem__)
    return QuestionAnswer(
        question_id="canonical_coverage_by_run",
        question="How many unique canonical suggestions did each run cover?",
        method="Map each raw suggestion to canonical ids and count uniques.",
        answer=(
            f"The best single run is {max_audit_id} with "
            f"{coverage_values[max_audit_id]} unique canonical suggestions."
        ),
        values=values,
    )


def _coverage_at_k(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> QuestionAnswer:
    by_source = _canonical_lookup(canonical)
    values: dict[str, object] = {}
    level_coverages: dict[str, dict[str, int]] = {}
    for level, audits in _level_groups(parsed).items():
        ordered = sorted(audits, key=lambda audit: audit.audit_id)
        level_values: dict[str, int] = {}
        seen: set[str] = set()
        for index, audit in enumerate(ordered, start=1):
            seen.update(_canonical_ids_for_audit(audit, by_source))
            level_values[f"k={index}"] = len(seen)
        level_coverages[level] = level_values
        values[level] = level_values

    minimal_k3 = level_coverages["minimal"]["k=3"]
    xhigh_k3 = level_coverages["xhigh"]["k=3"]
    return QuestionAnswer(
        question_id="coverage_at_k_by_level",
        question=(
            "At k=3, does minimal thinking produce the same canonical "
            "coverage as xhigh?"
        ),
        method=(
            "For each thinking level, union canonical ids over sorted runs "
            "audit_0 through audit_2."
        ),
        answer=(
            f"Minimal covers {minimal_k3} canonical suggestions at k=3; "
            f"xhigh covers {xhigh_k3}."
        ),
        values=values,
    )


def _cost_per_addressed(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> QuestionAnswer:
    by_source = _canonical_lookup(canonical)
    values: dict[str, float | None] = {}
    for audit in parsed.audits:
        addressed_ids = _canonical_ids_for_audit(
            audit, by_source, addressed_only=True
        )
        values[audit.audit_id] = (
            None
            if not addressed_ids
            else round(audit.stats.cost_usd / len(addressed_ids), 4)
        )
    priced_values = {
        audit_id: value
        for audit_id, value in values.items()
        if value is not None
    }
    best_audit_id = min(priced_values, key=priced_values.__getitem__)
    return QuestionAnswer(
        question_id="cost_per_unique_addressed_suggestion",
        question="What is the cost per unique addressed suggestion?",
        method=(
            "For each run, divide cost by the number of covered canonical "
            "suggestions whose resolution is not not_addressed."
        ),
        answer=(
            f"The lowest cost per unique addressed suggestion is "
            f"{best_audit_id} at ${values[best_audit_id]:.4f}."
        ),
        values=dict(values),
    )


def _repeated_not_addressed(
    canonical: CanonicalSuggestionSet,
) -> QuestionAnswer:
    values = {
        suggestion.canonical_id: len(suggestion.source_suggestions)
        for suggestion in canonical.canonical_suggestions
        if suggestion.resolution == Resolution.NOT_ADDRESSED
        and len(suggestion.source_suggestions) > 1
    }
    return QuestionAnswer(
        question_id="repeated_not_addressed_suggestions",
        question=(
            "Which suggestions are repeatedly produced but intentionally "
            "not addressed?"
        ),
        method=(
            "Filter canonical suggestions to not_addressed items with more "
            "than one source suggestion."
        ),
        answer=(
            f"{len(values)} not-addressed suggestions appeared in more than "
            "one audit."
        ),
        values=dict(sorted(values.items(), key=lambda item: item[0])),
    )


def _duplicate_pressure(canonical: CanonicalSuggestionSet) -> QuestionAnswer:
    counts = [
        len(suggestion.source_suggestions)
        for suggestion in canonical.canonical_suggestions
    ]
    values: dict[str, object] = {
        "mean_sources_per_canonical": _mean(
            [float(count) for count in counts]
        ),
        "max_sources_per_canonical": max(counts),
        "canonical_ids_with_max_sources": [
            suggestion.canonical_id
            for suggestion in canonical.canonical_suggestions
            if len(suggestion.source_suggestions) == max(counts)
        ],
    }
    return QuestionAnswer(
        question_id="duplicate_pressure",
        question="Which settings produce more duplicates?",
        method=(
            "Summarize how many raw suggestions map to each canonical "
            "suggestion across the full corpus."
        ),
        answer=(
            "The most repeated canonical suggestions have "
            f"{values['max_sources_per_canonical']} source suggestions."
        ),
        values=values,
    )


def _broad_suggestion_rate(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> QuestionAnswer:
    by_source = _canonical_lookup(canonical)
    values = {}
    for level, audits in _level_groups(parsed).items():
        raw_count = 0
        broad_count = 0
        for audit in audits:
            for suggestion in audit.suggestions:
                raw_count += 1
                if len(by_source[suggestion.suggestion_id]) > 1:
                    broad_count += 1
        values[level] = {
            "raw_suggestions": raw_count,
            "multi_canonical_suggestions": broad_count,
            "rate": round(broad_count / raw_count, 3),
        }
    return QuestionAnswer(
        question_id="broad_suggestion_rate_by_level",
        question="Which settings produce broader suggestions?",
        method=(
            "Count raw suggestions that map to more than one atomic "
            "canonical suggestion."
        ),
        answer="Broad-suggestion rates are reported by thinking level.",
        values=values,
    )


def _kind_mix(parsed: ParsedAuditSet) -> QuestionAnswer:
    values = {}
    for level, audits in _level_groups(parsed).items():
        counter: Counter[str] = Counter()
        for audit in audits:
            counter.update(audit.suggestion_kind_counts)
        values[level] = dict(sorted(counter.items()))
    return QuestionAnswer(
        question_id="suggestion_kind_mix_by_level",
        question="Which thinking level produced each kind of suggestion?",
        method="Sum parsed suggestion kinds across runs for each level.",
        answer="Suggestion kind counts are reported by thinking level.",
        values=values,
    )


def analyze(
    parsed: ParsedAuditSet, canonical: CanonicalSuggestionSet
) -> Analysis:
    _validate_canonical_suggestions(parsed, canonical)
    return Analysis(
        corpus_id=parsed.corpus_id,
        audit_count=len(parsed.audits),
        parsed_suggestion_count=sum(
            audit.suggestion_count for audit in parsed.audits
        ),
        canonical_suggestion_count=len(canonical.canonical_suggestions),
        questions=[
            _cost_comparison(parsed),
            _speed_comparison(parsed),
            _raw_suggestion_counts(parsed),
            _canonical_coverage(parsed, canonical),
            _coverage_at_k(parsed, canonical),
            _cost_per_addressed(parsed, canonical),
            _repeated_not_addressed(canonical),
            _duplicate_pressure(canonical),
            _broad_suggestion_rate(parsed, canonical),
            _kind_mix(parsed),
        ],
    )


def _write_json(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n")


def _write_analysis_markdown(path: Path, analysis: Analysis) -> None:
    lines = [
        "# Audit Corpus Ground Truth Analysis",
        "",
        f"- Audits: {analysis.audit_count}",
        f"- Parsed suggestions: {analysis.parsed_suggestion_count}",
        f"- Canonical suggestions: {analysis.canonical_suggestion_count}",
        "",
        "## Questions",
        "",
    ]
    for question in analysis.questions:
        lines.extend(
            [
                f"### {question.question_id}",
                "",
                f"**Question:** {question.question}",
                "",
                f"**Method:** {question.method}",
                "",
                f"**Answer:** {question.answer}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main(
    corpus_dir: Annotated[
        Path,
        typer.Option(
            "--corpus-dir",
            file_okay=False,
            dir_okay=True,
            help="Directory containing the audit corpus manifest.",
        ),
    ] = Path("data/audit-corpus"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Directory for curated and generated ground-truth artifacts.",
        ),
    ] = Path("data/audit-corpus/ground-truth"),
) -> None:
    parsed = parse_corpus(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "parsed_audits.json", parsed)

    canonical = _load_canonical_suggestions(output_dir)
    analysis = analyze(parsed, canonical)
    _write_json(output_dir / "analysis.json", analysis)
    _write_analysis_markdown(output_dir / "analysis.md", analysis)

    typer.echo(
        "Wrote parsed_audits.json, analysis.json, and analysis.md "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
