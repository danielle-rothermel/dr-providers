@AGENTS.md

## Live provider credentials

Run credentialed commands under `mise exec`; these keys should be served by
the repository's dotfiles-managed mise environment. Never print their values.

- OpenRouter uses `OPENROUTER_API_KEY` directly.
- Gemini uses `GEMINI_API_KEY` directly.
- OpenAI uses `MARIMO_OPENAI_API_KEY`; map it to `OPENAI_API_KEY` for
  dr-providers commands.
- Anthropic uses `OPENCODE_ANTHROPIC_API_KEY`; map it to `ANTHROPIC_API_KEY`
  for dr-providers commands.

Treat a missing key as a mise/dotfiles setup problem to diagnose before
concluding that live verification is unavailable.
