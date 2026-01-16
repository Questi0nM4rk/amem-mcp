# A-MEM MCP Server

Brain-like memory for Claude Code based on the NeurIPS 2025 paper "A-MEM: Agentic Memory for LLM Agents".

## Features

- **Automatic memory linking**: New memories automatically connect to related existing ones
- **Memory evolution**: New information updates the context of existing memories
- **Rich metadata**: Auto-generated keywords, context, and tags
- **Global storage**: Shared across all projects at `~/.codeagent/memory/`
- **Fallback mode**: Works without A-MEM library using simple keyword matching

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

## Philosophy

Based on [A-MEM](https://arxiv.org/abs/2409.07536) (2024) - "Agentic Memory for LLM Agents".

**Key insight**: Memory should behave like a brain, not a database. New memories automatically link to related ones (Zettelkasten-style), and existing memories evolve when new information arrives.

## Installation

### Basic (fallback mode - JSON storage)

```bash
pip install git+https://github.com/Questi0nM4rk/amem-mcp.git
```

### Full (ChromaDB + semantic search)

```bash
pip install "git+https://github.com/Questi0nM4rk/amem-mcp.git[full]"
```

Note: Full installation requires:
- `OPENAI_API_KEY` environment variable for metadata generation
- OR Ollama running locally

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
```

## How It Works

1. **Store**: Content is analyzed, keywords extracted, and similar memories found
2. **Link**: Bidirectional links created between related memories
3. **Evolve**: Existing memories' context updated with new information
4. **Search**: Vector similarity + link traversal for comprehensive results

## Backend

- **Full mode**: Uses A-MEM with ChromaDB for vector storage
- **Fallback mode**: JSON file with keyword-based matching (still functional!)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For full mode | LLM for metadata generation |
| `CODEAGENT_HOME` | Optional | Override default `~/.codeagent` |

## Storage

Memories stored at: `~/.codeagent/memory/`
- ChromaDB (full mode): `~/.codeagent/memory/chromadb/`
- JSON (fallback): `~/.codeagent/memory/memories.json`
