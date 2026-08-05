# dr-providers

[![CI](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dr-providers.svg)](https://pypi.org/project/dr-providers/)

| [Repo Definitions](https://danielle-rothermel.github.io/dr-providers/) | [dr-serialize v0.1.1](https://github.com/danielle-rothermel/dr-serialize) |
| --- | --- |

**dr-providers makes LLM provider calls through explicit, typed contracts.**
It supports OpenRouter, OpenAI, Gemini, and Anthropic and is organized into
these functional areas:

- **Call modeling** builds validated, identity-hashed definitions, configs,
  and requests from provider routes, generation controls, and transcripts.
- **Protocol translation** maps the shared call model into OpenAI Chat
  Completions, OpenAI Responses, OpenRouter, Gemini's OpenAI-compatible
  endpoint, and Anthropic Messages wire formats.
- **Transport execution** owns credentials, endpoints, timeout and retry
  policy, and bounded HTTP request dispatch.
- **Outcomes and evidence** represent expected successes and failures as typed
  data and preserve sanitized raw HTTP evidence in versioned invocation
  records.
- **Testing and access** provide a deterministic scripted provider, a Python
  API, a command-line interface, and an optional localhost HTTP facade.
