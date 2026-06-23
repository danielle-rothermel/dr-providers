# Agent Instructions

## Verification

- **Cursor agents:** Do not run `uv run pre-commit run --all-files` unless the
  user explicitly asks you to run checks or verification.
- **Other agents:** Run `uv run pre-commit run --all-files` after making code
  changes.
- Skip `uv run pre-commit run --all-files` after documentation-only or
  configuration-only changes (non-Cursor agents).
- When running checks, treat every issue reported by
  `uv run pre-commit run --all-files` as in scope to fix, even when the issue
  is outside the files you touched.


## Commits

- After completing a requested sequence of changes, commit the result unless
  the user specifically asks not to commit at the end.
- Use a single-line commit message.
- If your own changes are easy to isolate, commit only your own changes.
- If your own changes are not cleanly extractable, commit all changes in the
  files you touched.
- Before committing code changes, run `uv run pre-commit run --all-files`
  (non-Cursor agents, or when the user explicitly asked you to run checks).
- Before committing documentation-only or configuration-only changes, do not run
  `uv run pre-commit run --all-files`.

## Multi-Player Workflow

- Assume other agents and users may be operating in this repo at the same time.
- Never delete, revert, reset, or otherwise undo unexpected changes in the repo.
- Assume unexpected changes were made intentionally by the user or another
  process.
- If unexpected changes appear suspect or make the requested work difficult,
  ask how to proceed and describe the concern.
- Do not undo suspect changes while waiting for guidance.
