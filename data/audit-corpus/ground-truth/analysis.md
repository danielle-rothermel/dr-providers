# Audit Corpus Ground Truth Analysis

- Audits: 18
- Parsed suggestions: 167
- Canonical suggestions: 21

## Questions

### cost_off_vs_high

**Question:** How much cheaper is thinking off than high thinking?

**Method:** Compare mean run cost across the three off and high audits.

**Answer:** Thinking off averages $0.414, high averages $0.563; off is 26.5% cheaper.

### speed_low_vs_minimal

**Question:** How much faster is low thinking than minimal thinking?

**Method:** Compare mean throughput in tokens per second across the three low and minimal audits.

**Answer:** Low averages 151.0 tok/s, minimal averages 159.3 tok/s; low is 5.2% slower.

### raw_suggestion_counts

**Question:** How many raw suggestions did each run produce?

**Method:** Count parsed suggestion lines in each audit.

**Answer:** Runs produced between 7 and 12 raw suggestions.

### canonical_coverage_by_run

**Question:** How many unique canonical suggestions did each run cover?

**Method:** Map each raw suggestion to canonical ids and count uniques.

**Answer:** The best single run is gpt-5.5/high/audit_2 with 12 unique canonical suggestions.

### coverage_at_k_by_level

**Question:** At k=3, does minimal thinking produce the same canonical coverage as xhigh?

**Method:** For each thinking level, union canonical ids over sorted runs audit_0 through audit_2.

**Answer:** Minimal covers 10 canonical suggestions at k=3; xhigh covers 14.

### cost_per_unique_addressed_suggestion

**Question:** What is the cost per unique addressed suggestion?

**Method:** For each run, divide cost by the number of covered canonical suggestions whose resolution is not not_addressed.

**Answer:** The lowest cost per unique addressed suggestion is gpt-5.5/off/audit_2 at $0.0339.

### repeated_not_addressed_suggestions

**Question:** Which suggestions are repeatedly produced but intentionally not addressed?

**Method:** Filter canonical suggestions to not_addressed items with more than one source suggestion.

**Answer:** 5 not-addressed suggestions appeared in more than one audit.

### duplicate_pressure

**Question:** Which settings produce more duplicates?

**Method:** Summarize how many raw suggestions map to each canonical suggestion across the full corpus.

**Answer:** The most repeated canonical suggestions have 28 source suggestions.

### broad_suggestion_rate_by_level

**Question:** Which settings produce broader suggestions?

**Method:** Count raw suggestions that map to more than one atomic canonical suggestion.

**Answer:** Broad-suggestion rates are reported by thinking level.

### suggestion_kind_mix_by_level

**Question:** Which thinking level produced each kind of suggestion?

**Method:** Sum parsed suggestion kinds across runs for each level.

**Answer:** Suggestion kind counts are reported by thinking level.
