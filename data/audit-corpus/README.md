# Audit Corpus

A small, committed corpus of ponytail audit outputs used as the stable input
fixture for this repository's deterministic ground-truth generator and tests.
It contains three `gpt-5.5` audit runs at each thinking level for the prompt
recorded in `manifest.json`.

Treat the raw audit files and curated canonical mapping as stable inputs. The
parsed results and analysis are committed generator outputs verified by the
test suite.

## Layout

```text
data/audit-corpus/
  README.md
  manifest.json          # corpus metadata: model, prompt, levels, file index
  gpt-5.5/               # raw audit markdown, one dir per thinking level
    off/                 #   audit_0.md, audit_1.md, audit_2.md
    minimal/             #   (same three files per level)
    low/
    medium/
    high/
    xhigh/
  ground-truth/          # curated + generated normalization of the raw audits
    canonical_suggestions.json
    parsed_audits.json
    analysis.json
    analysis.md
```

- `manifest.json`: corpus id, source repo, agent/model, prompt, thinking
  levels, and the index of raw audit files with stable `audit_id`s.
- `gpt-5.5/<level>/audit_*.md`: raw audit output, including the run stats
  header and the suggestion lines.
- `ground-truth/canonical_suggestions.json`: hand-curated. Defines the atomic
  unique suggestions, maps each raw audit suggestion to one or more canonical
  ids, and records the final resolution decision.
- `ground-truth/parsed_audits.json`: generated deterministic parse of the raw
  audit markdown (run metadata, suggestion text, file references, kind counts,
  net estimates).
- `ground-truth/analysis.json` / `analysis.md`: generated answers to the
  first-pass benchmark questions, machine- and human-readable.

## Regeneration

Regenerate the deterministic ground-truth artifacts after changing the parser,
the raw corpus, or the canonical mapping:

```bash
uv run python scripts/generate_audit_ground_truth.py \
  --corpus-dir data/audit-corpus \
  --output-dir data/audit-corpus/ground-truth
```

(Both options default to these paths, so a bare
`uv run python scripts/generate_audit_ground_truth.py` is equivalent.)

The generator overwrites `parsed_audits.json`, `analysis.json`, and
`analysis.md`. It validates `canonical_suggestions.json` but never overwrites
it — the canonical grouping and resolution labels are the curated ground
truth. The raw `gpt-5.5/` audit files are one-time captures and have no
regeneration script.
