# Git hooks

One-time setup:

```bash
git config core.hooksPath .githooks
```

## `pre-commit`

Runs `scripts/commit_guard.py --mode fast` — the three fast gates from the
Commit Strategy:

- `staged-diff inspection` (never-commit paths, size limit)
- `ruff check`
- `ruff format --check`

Typical runtime: ~5 seconds.

Slower gates (`mypy`, `pytest`, `benchmark`) are the agent's responsibility
via `commit_guard.py --mode full` or `--mode task` before it stages files.

## Bypass (interactive only)

```bash
SKIP_COMMIT_GUARD=1 git commit ...
```

Autonomous agents must never set this variable.
