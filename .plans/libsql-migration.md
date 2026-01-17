# Plan: Migrate to libSQL

## Changes

Replace ChromaDB/JSON fallback with libSQL vector search.

### Files to Modify

- `src/amem_mcp/server.py`:
  - Remove ChromaDB imports and fallback logic
  - Add libSQL with `F32_BLOB(384)` for embeddings
  - Use `vector_distance_cos()` for similarity search
  - Keep auto-linking logic (keyword + vector similarity)

### Dependencies

```toml
[project.dependencies]
libsql-experimental = ">=0.0.50"
sentence-transformers = ">=2.2.0"
```

### Database Location

`~/.codeagent/codeagent.db` (shared with other MCPs)

### Schema

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT,         -- JSON array
    context TEXT,
    tags TEXT,             -- JSON array
    links TEXT,            -- JSON array
    project TEXT,
    embedding F32_BLOB(384),
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS libsql_vector_idx_memories ON memories(embedding);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
```

### Embedding Model

Use `all-MiniLM-L6-v2` (384 dimensions) from sentence-transformers.

### Key Implementation Notes

1. **Auto-linking**: When storing a memory, search for similar memories and add bidirectional links
2. **Memory evolution**: When a related memory is found, update its context with new information
3. **Keyword extraction**: Use spaCy for NLP-based keyword extraction
4. **Fallback**: If sentence-transformers fails to load, use keyword-only matching

### Verification

- `uv run pytest tests/ -v`
- Manual: store_memory, search_memory (check vector similarity works)
