from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from dr_providers.names import ProviderName
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
                base_url=ProviderName.OPENROUTER.api_base_url,
                api_key_env=ProviderName.OPENROUTER.api_key_env(),
            ),
            client=client,
        )
