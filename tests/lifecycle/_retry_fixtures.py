from __future__ import annotations

from dr_providers.lifecycle import (
    CustomProviderCallRetryPolicy,
    ProviderInvocationOutcome,
)


def two_invocation_transient_retry_policy() -> CustomProviderCallRetryPolicy:
    return CustomProviderCallRetryPolicy(
        maximum_invocations=2,
        eligible_outcomes=frozenset(
            {
                ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
                ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
            }
        ),
        declared_delays_seconds=(1.0,),
    )
