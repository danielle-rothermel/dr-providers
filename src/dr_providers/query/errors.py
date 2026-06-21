class ProviderError(Exception):
    pass


class ProviderTransportError(ProviderError):
    pass


class ProviderSemanticError(ProviderError):
    pass
