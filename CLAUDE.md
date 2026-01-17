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
