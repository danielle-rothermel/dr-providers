"""Serve-facade public API (FastAPI app lives behind the [serve] extra)."""

from dr_providers.serve.runner import (
    ModelVariance,
    QueryResult,
    QuerySpec,
    ServeProviderKind,
    VarianceRecord,
    VarianceReport,
    build_request,
    run_query,
    run_variance,
)

__all__ = [
    "ModelVariance",
    "QueryResult",
    "QuerySpec",
    "ServeProviderKind",
    "VarianceRecord",
    "VarianceReport",
    "build_request",
    "run_query",
    "run_variance",
]
