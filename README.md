# dr-providers

OpenRouter LLM query client with typed requests and responses. Requires Python 3.13+.

## Install

```bash
pip install dr-providers
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dr-providers
```

### Optional CLI

```bash
pip install "dr-providers[cli]"
query-provider --help
```

## Authentication

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Library usage

```python
from dr_providers import (
    LlmRequest,
    Message,
    MessageRole,
    OpenRouterProvider,
    ProviderName,
)

with OpenRouterProvider() as provider:
    response = provider.generate(
        LlmRequest(
            provider=ProviderName.OPENROUTER,
            model="openai/gpt-4o-mini",
            messages=[
                Message(role=MessageRole.USER, content="Say hello in one word."),
            ],
        )
    )
    print(response.text)
```

### Prompt helper

For scripts and demos, `query_from_prompt` builds a request from plain strings:

```python
from dr_providers import OpenRouterProvider, ProviderName
from dr_providers.query.from_prompt import query_from_prompt

with OpenRouterProvider() as provider:
    response = query_from_prompt(
        provider,
        ProviderName.OPENROUTER,
        model="openai/gpt-4o-mini",
        prompt="Say hello in one word.",
    )
```

This helper is not re-exported from the top-level package.

## Public API

Import stable symbols from the top-level package:

```python
from dr_providers import LlmRequest, OpenRouterProvider, ReasoningSpec
```

See `dr_providers.__all__` for the full list.

## Development

```bash
uv sync
scripts/pre-check.sh
```

Run the CLI from the repo without installing:

```bash
uv run python scripts/query_provider.py --model openai/gpt-4o-mini -m "hi"
```
