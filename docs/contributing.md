# Git hooks

Local git hooks are optional but recommended.

## Enable hooks
```bash
# from repo root
git config core.hooksPath .githooks
```

## Pre-push
Runs `scripts/check.sh` (ruff, mypy, pytest) before pushing.