from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from dr_providers.names import (
    OPENROUTER_API_BASE_URL,
    OPENROUTER_API_KEY_ENV,
    ProviderName,
)
from dr_providers.query.transport import ApiProvider
from dr_providers.query.transport_config import ProviderConfig


class OpenRouterProvider(ApiProvider):
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=ProviderConfig(
                name=ProviderName.OPENROUTER,
                base_url=OPENROUTER_API_BASE_URL,
                api_key_env=OPENROUTER_API_KEY_ENV,
            ),
            client=client,
        )
