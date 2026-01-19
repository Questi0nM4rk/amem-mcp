# CLAUDE.md

A-MEM MCP Server - Brain-like memory with automatic linking and evolution for Claude Code.

## Structure

- `src/amem_mcp/server.py` - Main MCP server (all tools)
- `src/amem_mcp/__init__.py` - Package entry point
- `pyproject.toml` - Dependencies and build config

## Development

```bash
# Install with uv
uv pip install -e ".[full]"

# Run tests
uv run pytest tests/ -v

# Format/lint
uv run ruff format .
uv run ruff check .

# Run server
uv run amem-mcp
```

## Database

libSQL at `~/.codeagent/codeagent.db` with 384-dim sentence-transformer embeddings.

## Git Workflow

**IMPORTANT**: Direct push to `main` is disabled. All changes must go through PRs.

1. Create a feature branch: `git checkout -b feat/description`
2. Make changes and commit
3. Push branch: `git push -u origin feat/description`
4. Create PR via `gh pr create` or GitHub UI
5. CodeRabbit provides automated review
6. After addressing comments, **resolve each GitHub conversation**
7. Merge when approved

### CodeRabbit Tips

- Push fixes for review comments
- Click "Resolve conversation" on each addressed comment
- CodeRabbit auto-approves when all resolved and checks pass
