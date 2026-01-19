# A-MEM MCP Server

Brain-like memory for Claude Code based on the NeurIPS 2025 paper "A-MEM: Agentic Memory for LLM Agents".

## Features

- **Automatic memory linking**: New memories automatically connect to related existing ones
- **Memory evolution**: New information updates the context of existing memories
- **Rich metadata**: Auto-generated keywords, context, and tags using spaCy NLP
- **Vector search**: 384-dimension sentence-transformer embeddings for semantic similarity
- **Global storage**: Shared libSQL database at `~/.codeagent/codeagent.db`

## Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store knowledge with automatic linking and evolution |
| `search_memory` | Semantic search across all memories |
| `read_memory` | Read specific memory by ID with full metadata |
| `list_memories` | List recent memories with filtering |
| `update_memory` | Update existing memory (triggers re-evolution) |
| `delete_memory` | Remove a memory |
| `get_memory_stats` | Statistics about the memory system |
| `evolve_now` | Manually trigger memory evolution/consolidation |

## Philosophy

Based on [A-MEM](https://arxiv.org/abs/2409.07536) (2024) - "Agentic Memory for LLM Agents".

**Key insight**: Memory should behave like a brain, not a database. New memories automatically link to related ones (Zettelkasten-style), and existing memories evolve when new information arrives.

## Installation

### Basic (keyword matching only)

```bash
pip install git+https://github.com/Questi0nM4rk/amem-mcp.git
```

### Full (vector search + NLP keywords)

```bash
pip install "git+https://github.com/Questi0nM4rk/amem-mcp.git[full]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage with Claude Code

```bash
claude mcp add amem -- python -m amem_mcp.server
```

## Usage

```python
# Store a memory
store_memory(content="JaCore uses repository pattern with Unit of Work")

# Search memories
search_memory(query="data access patterns")

# Read specific memory
read_memory(memory_id="mem_0001")

# Filter by project
search_memory(query="architecture", project="JaCore")
```

## How It Works

1. **Store**: Content is analyzed, keywords extracted (spaCy NLP), and similar memories found
2. **Link**: Bidirectional links created between related memories
3. **Evolve**: Existing memories' context updated with new information
4. **Search**: Vector similarity + keyword matching for comprehensive results

## Backend

| Feature | Full Mode | Basic Mode |
|---------|-----------|------------|
| Storage | libSQL | libSQL |
| Search | Vector similarity | Keyword matching |
| Keywords | spaCy NLP | Basic tokenization |
| Dependencies | sentence-transformers, spacy | None |

- **Full mode**: Uses sentence-transformers for vector embeddings with spaCy for NLP keyword extraction
- **Basic mode**: Falls back to keyword-based matching (still functional!)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CODEAGENT_HOME` | Optional | Override default `~/.codeagent` |

## Storage

Database: `~/.codeagent/codeagent.db` (libSQL with vector search)

Schema:
- `memories` table with 384-dimension F32_BLOB embeddings
- Automatic vector index for similarity search
- JSON fields for keywords, tags, and links

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[full]" --group dev

# Run tests
uv run pytest tests/ -v

# Format and lint
uv run ruff format .
uv run ruff check .
```
