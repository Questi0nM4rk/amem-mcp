# Code Review Workflow

## Branch Protection

**Direct pushes to `main` are disabled.** All changes must go through pull requests.

## Pre-Commit Checks

This repository includes a pre-commit hook that runs before every commit:

```bash
# Install the pre-commit hook
git config core.hooksPath .githooks
```

The hook runs:
1. `ruff check src/ tests/` - Python linting
2. `ruff format --check src/ tests/` - Python format verification
3. `pyright src/ tests/` - Type checking
4. `pytest tests/ -v` - Unit tests
5. `markdownlint-cli2 "**/*.md"` - Markdown linting

All checks must pass before a commit is accepted.

**Why strict?** These MCPs are primarily developed by AI agents. Strict guardrails catch issues locally before they waste PR review cycles.

## Pull Request Process

### Before Creating a PR

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/description
   ```

2. Make changes and commit (pre-commit hook runs automatically)

3. Push to remote:
   ```bash
   git push -u origin feat/description
   ```

### During PR Review

1. **CodeRabbit** provides automated AI review
2. Address all actionable comments
3. **Verify fixes manually** - CodeRabbit's `@coderabbitai resolve` does NOT validate fixes

### Before Merging

1. Run full validation locally:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run pyright src/ tests/
   uv run pytest tests/ -v
   ```

2. Request re-review if significant changes were made:
   ```
   @coderabbitai review
   ```

3. Only resolve CodeRabbit comments after verifying fixes:
   - Manual: Click "Resolve conversation" on each verified fix
   - Bulk (use with caution): `@coderabbitai resolve`

4. Ensure CI passes (GitHub Actions)

5. Get approval and merge

## CI Pipeline

GitHub Actions runs on every push and PR:
- Linting (ruff check)
- Format check (ruff format --check)
- Type check (pyright)
- Tests (pytest) on Python 3.10, 3.11, 3.12

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv run ruff check .` | Run Python linter |
| `uv run ruff format .` | Auto-format Python code |
| `uv run pyright src/ tests/` | Type check |
| `uv run pytest tests/ -v` | Run tests |
| `npx markdownlint-cli2 "**/*.md"` | Lint markdown files |
| `@coderabbitai review` | Request fresh review |
| `@coderabbitai resolve` | Bulk resolve (use carefully) |
