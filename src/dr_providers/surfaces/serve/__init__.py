"""Serve-facade public API (FastAPI app lives behind the [serve] extra)."""

from dr_providers.surfaces.serve.query import (
    QueryResult,
    QuerySpec,
    ServeProviderKind,
    build_request,
    run_query,
)
from dr_providers.surfaces.serve.variance import (
    ModelVariance,
    VarianceRecord,
    VarianceReport,
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
