"""Provider Call Definition: historical import path.

The Provider Call Definition and Provider Call Config models are
co-located in :mod:`dr_providers.config` so materialization is a plain
method with no circular import. This module re-exports the Definition and
its schema constants for their original import path.
"""

from __future__ import annotations

from dr_providers.config import (
    PROVIDER_CALL_DEFINITION_SCHEMA,
    PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
    ProviderCallDefinition,
)

__all__ = [
    "PROVIDER_CALL_DEFINITION_SCHEMA",
    "PROVIDER_CALL_DEFINITION_SCHEMA_VERSION",
    "ProviderCallDefinition",
]
