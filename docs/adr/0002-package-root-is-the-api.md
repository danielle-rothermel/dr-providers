# The package root is the canonical API; transport imports stay lazy

With the 0.1.x `query` lineage deleted (it had zero external importers —
verified against unitbench and whetstone-ai, 2026-07), `from dr_providers
import ...` becomes the single canonical Python surface, replacing
`dr_providers.kernel`. whetstone-ai's ~19 `dr_providers.kernel` import sites
migrate in lockstep; no alias module remains, because two blessed paths for
every symbol is the duplicate-surface problem this cleanup exists to kill.

One constraint is invisible in the code and must survive any future
restructuring: importing the failure taxonomy (or any pure module — configs,
payloads) must never pull in httpx. whetstone's platform imports
`FailureClass`/`ProviderFailure` under an import-hygiene contract. The root
therefore re-exports transport symbols (`HttpProvider`, `TransportPolicy`)
through a lazy `__getattr__`, not a plain import.
