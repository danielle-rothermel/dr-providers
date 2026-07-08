# The provider-call seam is synchronous

`Provider.complete()` is sync, deliberately, even though LLM SDKs are
async-first. Parallelism in this ecosystem lives caller-side in sync-shaped
machinery: DBOS queues (durable fan-out across steps), DSPy thread pools, and
plain thread pools over the thread-safe httpx client. Going async would
virally recolor every whetstone call site during a hardening phase, and
in-process asyncio fan-out inside a durable DBOS step would blur the
durability boundary queues already handle.

This stays cheap to revisit because the kernel keeps all hard logic —
payload building, response parsing, failure taxonomy, conformance — in pure,
color-agnostic functions. The only colored code is the ~100-line httpx
transport; an `AsyncHttpProvider` twin over the same pure kernel is the
escape hatch if a single-process fan-out wall ever appears. Subinterpreters,
free-threading, and multiprocessing were considered for the wider ecosystem
and rejected for this repo: LLM calls are pure I/O with no compute to
parallelize (and PyO3-based Rust extensions cannot load in subinterpreters,
which also rules that path out for the compute-heavy repos).
