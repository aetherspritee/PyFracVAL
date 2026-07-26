# Archived benchmark configs

These orchestrator configs probe CCA sticking features that were measured and
found **not** to improve on the vanilla Fibonacci baseline (or, in the case of
the `retry_mode_matrix*`/`benchmark_comparison_normal` files, features that
were later confirmed to make no measurable difference). See
`docs/source/experiments.md` for the data and reasoning.

Kept for reference rather than deleted — they're runnable if someone wants to
reproduce or extend the comparison. The `plausibility_step2_*` variants here
(`*.smoke.toml`, `*_fast.toml`, `*_missing_only.toml`, `batches/`) are
one-off/derived shards of the canonical `configs/plausibility_step2_feature_matrix.toml`,
archived for the same "keep it, don't need it in the main list" reason rather
than because the feature they test lost.
