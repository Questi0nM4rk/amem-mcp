"""
A-MEM MCP Server - Brain-like Memory for Claude Code

Based on NeurIPS 2025 paper "A-MEM: Agentic Memory for LLM Agents"
Implements Zettelkasten-inspired memory with dynamic linking and evolution.

Features:
- Persistent libSQL storage with vector search
- Automatic memory linking (new memories connect to related existing ones)
- Memory evolution (new info updates existing memory context)
- Rich metadata generation (keywords, context, tags)
- Global memory shared across all projects

Database: ~/.codeagent/codeagent.db
"""

import json
import logging
import os
import struct
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import libsql_experimental as libsql
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
CODEAGENT_DIR = Path(os.environ.get("CODEAGENT_HOME", Path.home() / ".codeagent"))
CODEAGENT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CODEAGENT_DIR / "codeagent.db"

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Initialize FastMCP server
mcp = FastMCP(
    "amem",
    instructions="Brain-like memory for Claude Code using A-MEM architecture. "
    "Memories automatically link to each other and evolve over time. "
    "Use this for storing and retrieving project knowledge, patterns, and decisions.",
)

# Lazy-loaded singletons
_embedding_model = None
_nlp = None


def _get_embedding_model():
    """Get or load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except ImportError:
        logger.warning("sentence-transformers not installed, vector search disabled")
        _embedding_model = None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        _embedding_model = None

    return _embedding_model


def _get_nlp():
    """Get or load spaCy NLP pipeline with graceful fallback."""
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm model")
        except OSError:
            logger.info("Downloading spaCy en_core_web_sm model...")
            from spacy.cli import download

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    except ImportError:
        logger.debug("spaCy not installed, using basic keyword extraction")
        _nlp = None
    except Exception as e:
        logger.warning(f"Failed to load spaCy: {e}")
        _nlp = None

    return _nlp


@contextmanager
def _get_db() -> Generator[libsql.Connection, None, None]:
    """Get database connection, creating schema if needed. Closes on exit."""
    conn = libsql.connect(str(DB_PATH))
    _init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def _init_schema(conn: libsql.Connection) -> None:
    """Initialize database schema."""
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT,
            context TEXT,
            tags TEXT,
            links TEXT,
            project TEXT,
            embedding F32_BLOB({EMBEDDING_DIM}),
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
        CREATE INDEX IF NOT EXISTS idx_memories_memory_id ON memories(memory_id);

        CREATE TABLE IF NOT EXISTS memory_counter (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            counter INTEGER DEFAULT 0
        );

        INSERT OR IGNORE INTO memory_counter (id, counter) VALUES (1, 0);
    """)

    # Create vector index if not exists (separate statement)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS libsql_vector_idx_memories ON memories(embedding)"
        )
    except Exception as e:
        logger.debug(f"Vector index may already exist: {e}")

    conn.commit()


def _embed(text: str) -> bytes | None:
    """Generate embedding for text, returns F32_BLOB bytes."""
    model = _get_embedding_model()
    if model is None:
        return None

    try:
        embedding = model.encode(text, convert_to_numpy=True)
        # Pack as F32_BLOB (little-endian floats)
        return struct.pack(f"<{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.exception(f"Embedding generation failed: {e}")
        return None


def _json_loads(val: str | None) -> list[str]:
    """Parse JSON array or return empty list."""
    if val is None:
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _json_dumps(val: list[str] | None) -> str:
    """Serialize list to JSON."""
    return json.dumps(val or [])


# Stopwords for keyword extraction
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "also",
    "use",
    "used",
    "using",
    "uses",
    "get",
    "gets",
    "got",
    "make",
    "makes",
    "made",
}


def _extract_keywords(text: str) -> list[str]:
    """Extract keywords using spaCy or fallback to basic extraction."""
    nlp = _get_nlp()

    if nlp is not None:
        return _extract_keywords_spacy(text, nlp)
    else:
        return _extract_keywords_basic(text)


def _extract_keywords_spacy(text: str, nlp) -> list[str]:
    """Extract keywords using spaCy."""
    doc = nlp(text[:100000])

    keywords = []
    seen_lemmas = set()

    relevant_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
    for token in doc:
        if (
            token.pos_ in relevant_pos
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 2
            and token.is_alpha
        ):
            lemma = token.lemma_.lower()
            if lemma not in seen_lemmas and lemma not in STOPWORDS:
                keywords.append(lemma)
                seen_lemmas.add(lemma)

    entity_labels = {"ORG", "PRODUCT", "GPE", "WORK_OF_ART", "LAW"}
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            ent_text = ent.text.lower()
            if ent_text not in seen_lemmas and len(ent_text) > 2:
                keywords.append(ent_text)
                seen_lemmas.add(ent_text)

    return keywords[:20]


def _extract_keywords_basic(text: str) -> list[str]:
    """Fallback: basic keyword extraction without spaCy."""
    words = text.lower().split()
    keywords = []
    seen = set()

    for word in words:
        cleaned = word.strip(".,!?;:()[]{}\"'-")
        if (
            len(cleaned) > 2
            and cleaned not in STOPWORDS
            and cleaned.isalnum()
            and cleaned not in seen
        ):
            keywords.append(cleaned)
            seen.add(cleaned)

    return keywords[:15]


def _generate_context(keywords: list[str]) -> str:
    """Generate context for a memory."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"Memory added on {timestamp}. Keywords: {', '.join(keywords[:5])}"


def _find_links(
    conn: libsql.Connection,
    keywords: list[str],
    content: str,
    exclude_id: str | None = None,
) -> list[str]:
    """Find related memories using vector similarity and keyword overlap."""
    links = []

    # Try vector search first
    embedding = _embed(content)
    if embedding is not None:
        try:
            cursor = conn.execute(
                """
                SELECT memory_id, keywords,
                       vector_distance_cos(embedding, ?) as distance
                FROM memories
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT 10
                """,
                (embedding,),
            )
            rows = cursor.fetchall()

            keyword_set = set(kw.lower() for kw in keywords)
            for row in rows:
                mem_id, stored_keywords_json, distance = row
                if exclude_id and mem_id == exclude_id:
                    continue

                stored_keywords = _json_loads(stored_keywords_json)
                stored_set = set(kw.lower() for kw in stored_keywords)
                overlap = len(keyword_set & stored_set)

                # Link if good vector similarity OR keyword overlap
                if distance < 0.5 or overlap >= 2:
                    links.append(mem_id)

                if len(links) >= 5:
                    break

        except Exception as e:
            logger.debug(f"Vector search failed, falling back to keywords: {e}")

    # Fallback: keyword-only search
    if not links:
        try:
            cursor = conn.execute(
                "SELECT memory_id, keywords FROM memories ORDER BY created_at DESC LIMIT 50"
            )
            keyword_set = set(kw.lower() for kw in keywords)

            for row in cursor.fetchall():
                mem_id, stored_keywords_json = row
                if exclude_id and mem_id == exclude_id:
                    continue

                stored_keywords = _json_loads(stored_keywords_json)
                stored_set = set(kw.lower() for kw in stored_keywords)
                overlap = len(keyword_set & stored_set)

                if overlap >= 2:
                    links.append(mem_id)

                if len(links) >= 5:
                    break

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")

    return links


# ============================================
# MCP Tools
# ============================================


@mcp.tool()
def store_memory(
    content: str,
    tags: list[str] | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """
    Store knowledge with automatic linking and memory evolution.

    The memory system will:
    1. Auto-generate keywords, context, and tags
    2. Find related existing memories and link them
    3. Update existing memories with new context (evolution)

    Args:
        content: The knowledge to store (patterns, decisions, insights)
        tags: Optional tags for categorization (e.g., ["architecture", "pattern"])
        project: Optional project name for filtering later

    Returns:
        Stored memory ID with generated metadata and links
    """
    try:
        with _get_db() as conn:
            # Generate memory ID atomically
            cursor = conn.execute(
                "UPDATE memory_counter SET counter = counter + 1 WHERE id = 1 RETURNING counter"
            )
            row = cursor.fetchone()
            memory_id = f"mem_{row[0]:04d}"

            # Extract metadata
            keywords = _extract_keywords(content)
            context = _generate_context(keywords)

            # Find related memories for linking
            links = _find_links(conn, keywords, content, exclude_id=memory_id)

            # Add project as tag if provided
            all_tags = list(tags) if tags else []
            if project:
                all_tags.append(f"project:{project}")

            # Generate embedding
            embedding = _embed(content)

            now = datetime.now().isoformat()

            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, content, keywords, context, tags, links,
                    project, embedding, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    content,
                    _json_dumps(keywords),
                    context,
                    _json_dumps(all_tags),
                    _json_dumps(links),
                    project,
                    embedding,
                    now,
                    now,
                ),
            )

            # Add reverse links to linked memories
            for linked_id in links:
                cursor = conn.execute(
                    "SELECT links FROM memories WHERE memory_id = ?",
                    (linked_id,),
                )
                row = cursor.fetchone()
                if row:
                    existing_links = _json_loads(row[0])
                    if memory_id not in existing_links:
                        existing_links.append(memory_id)
                        conn.execute(
                            "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                            (_json_dumps(existing_links[:10]), now, linked_id),
                        )

            conn.commit()

            return {
                "stored": True,
                "memory_id": memory_id,
                "content_preview": content[:100] + "..."
                if len(content) > 100
                else content,
                "keywords": keywords,
                "context": context,
                "linked_to": len(links),
                "storage_path": str(DB_PATH),
            }

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        return {"stored": False, "error": str(e)}


@mcp.tool()
def search_memory(
    query: str,
    k: int = 5,
    project: str | None = None,
) -> dict[str, Any]:
    """
    Semantic search across all memories.

    Finds memories related to your query using vector similarity
    and the Zettelkasten link structure.

    Args:
        query: What to search for (natural language)
        k: Maximum results to return (default 5)
        project: Optional filter by project name

    Returns:
        Relevant memories with content, keywords, context, and links
    """
    try:
        with _get_db() as conn:
            k = min(max(k, 1), 20)

            results = []

            # Try vector search first
            embedding = _embed(query)
            if embedding is not None:
                try:
                    sql = """
                        SELECT memory_id, content, keywords, context, tags, links,
                               vector_distance_cos(embedding, ?) as distance
                        FROM memories
                        WHERE embedding IS NOT NULL
                    """
                    params: list[Any] = [embedding]

                    if project:
                        sql += " AND project = ?"
                        params.append(project)

                    sql += " ORDER BY distance ASC LIMIT ?"
                    params.append(k)

                    cursor = conn.execute(sql, params)

                    for row in cursor.fetchall():
                        (
                            mem_id,
                            content,
                            kw_json,
                            ctx,
                            tags_json,
                            links_json,
                            distance,
                        ) = row
                        score = 1.0 / (1.0 + distance) if distance else 1.0
                        results.append(
                            {
                                "id": mem_id,
                                "content": content,
                                "keywords": _json_loads(kw_json),
                                "context": ctx,
                                "tags": _json_loads(tags_json),
                                "links": _json_loads(links_json),
                                "relevance": round(score, 3),
                            }
                        )

                except Exception as e:
                    logger.debug(f"Vector search failed: {e}")

            # Fallback: keyword search
            if not results:
                query_keywords = set(_extract_keywords(query))

                sql = "SELECT memory_id, content, keywords, context, tags, links FROM memories"
                params = []

                if project:
                    sql += " WHERE project = ?"
                    params.append(project)

                sql += " LIMIT 100"

                cursor = conn.execute(sql, params)

                scored = []
                for row in cursor.fetchall():
                    mem_id, content, kw_json, ctx, tags_json, links_json = row
                    mem_keywords = set(_json_loads(kw_json))
                    overlap = len(query_keywords & mem_keywords)
                    content_score = sum(
                        1 for kw in query_keywords if kw in content.lower()
                    )
                    score = overlap * 0.6 + content_score * 0.4

                    if score > 0 or not query.strip():
                        scored.append(
                            (
                                score if query.strip() else 1.0,
                                {
                                    "id": mem_id,
                                    "content": content,
                                    "keywords": list(mem_keywords),
                                    "context": ctx,
                                    "tags": _json_loads(tags_json),
                                    "links": _json_loads(links_json),
                                },
                            )
                        )

                scored.sort(key=lambda x: x[0], reverse=True)
                results = [
                    {**mem, "relevance": round(score, 3)} for score, mem in scored[:k]
                ]

            # Note: project filtering is already done in SQL for both vector and keyword paths

            return {
                "query": query,
                "results": results[:k],
                "count": len(results[:k]),
            }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"query": query, "results": [], "count": 0, "error": str(e)}


@mcp.tool()
def read_memory(memory_id: str) -> dict[str, Any]:
    """
    Read a specific memory by ID with full metadata.

    Args:
        memory_id: The memory ID to retrieve

    Returns:
        Full memory content with keywords, context, tags, and links
    """
    try:
        with _get_db() as conn:
            cursor = conn.execute(
                """
                SELECT memory_id, content, keywords, context, tags, links, created_at, updated_at
                FROM memories WHERE memory_id = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()

            if not row:
                return {"found": False, "error": f"Memory not found: {memory_id}"}

            return {
                "found": True,
                "id": row[0],
                "content": row[1],
                "keywords": _json_loads(row[2]),
                "context": row[3],
                "tags": _json_loads(row[4]),
                "links": _json_loads(row[5]),
                "created_at": row[6],
                "updated_at": row[7],
            }

    except Exception as e:
        return {"found": False, "error": str(e)}


@mcp.tool()
def list_memories(
    limit: int = 10,
    project: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """
    List recent memories with optional filtering.

    Args:
        limit: Maximum memories to return (default 10)
        project: Filter by project name
        tag: Filter by tag

    Returns:
        List of recent memories
    """
    try:
        with _get_db() as conn:
            limit = min(max(limit, 1), 50)

            sql = """
                SELECT memory_id, content, keywords, tags
                FROM memories
                WHERE 1=1
            """
            params: list[Any] = []

            if project:
                sql += " AND project = ?"
                params.append(project)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit * 2)  # Get more to account for tag filtering

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            memories = []
            for row in rows:
                mem_id, content, kw_json, tags_json = row
                tags_list = _json_loads(tags_json)

                # Filter by tag if specified
                if tag and tag not in tags_list:
                    continue

                memories.append(
                    {
                        "id": mem_id,
                        "content_preview": content[:100],
                        "keywords": _json_loads(kw_json)[:5],
                        "tags": tags_list,
                    }
                )

                if len(memories) >= limit:
                    break

            return {
                "memories": memories,
                "count": len(memories),
                "filters": {"project": project, "tag": tag},
            }

    except Exception as e:
        return {"memories": [], "count": 0, "error": str(e)}


@mcp.tool()
def update_memory(
    memory_id: str,
    content: str,
) -> dict[str, Any]:
    """
    Update an existing memory. Triggers re-evolution of links.

    Args:
        memory_id: The memory to update
        content: New content

    Returns:
        Updated memory details
    """
    try:
        with _get_db() as conn:
            # Get existing links for bidirectional consistency
            cursor = conn.execute(
                "SELECT links FROM memories WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if not row:
                return {"updated": False, "error": f"Memory not found: {memory_id}"}

            old_links = set(_json_loads(row[0]))

            # Re-extract metadata
            keywords = _extract_keywords(content)
            context = _generate_context(keywords)
            new_links = _find_links(conn, keywords, content, exclude_id=memory_id)
            new_links_set = set(new_links)
            embedding = _embed(content)
            now = datetime.now().isoformat()

            conn.execute(
                """
                UPDATE memories SET
                    content = ?, keywords = ?, context = ?, links = ?,
                    embedding = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (
                    content,
                    _json_dumps(keywords),
                    context,
                    _json_dumps(new_links),
                    embedding,
                    now,
                    memory_id,
                ),
            )

            # Update bidirectional links: add reverse links to new targets
            added = new_links_set - old_links
            for linked_id in added:
                cursor = conn.execute(
                    "SELECT links FROM memories WHERE memory_id = ?",
                    (linked_id,),
                )
                link_row = cursor.fetchone()
                if link_row:
                    existing = _json_loads(link_row[0])
                    if memory_id not in existing:
                        existing.append(memory_id)
                        conn.execute(
                            "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                            (_json_dumps(existing[:10]), now, linked_id),
                        )

            # Remove reverse links from removed targets
            removed = old_links - new_links_set
            for linked_id in removed:
                cursor = conn.execute(
                    "SELECT links FROM memories WHERE memory_id = ?",
                    (linked_id,),
                )
                link_row = cursor.fetchone()
                if link_row:
                    existing = _json_loads(link_row[0])
                    if memory_id in existing:
                        existing.remove(memory_id)
                        conn.execute(
                            "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                            (_json_dumps(existing), now, linked_id),
                        )

            conn.commit()

            return {
                "updated": True,
                "memory_id": memory_id,
                "new_keywords": keywords,
                "new_links": len(new_links),
            }

    except Exception as e:
        return {"updated": False, "error": str(e)}


@mcp.tool()
def delete_memory(memory_id: str) -> dict[str, Any]:
    """
    Delete a memory.

    Args:
        memory_id: The memory to delete

    Returns:
        Confirmation of deletion
    """
    try:
        with _get_db() as conn:
            now = datetime.now().isoformat()

            # Remove backlinks from other memories that link to this one
            # LIKE is a pre-filter; exact match is verified in Python after JSON parse
            cursor = conn.execute(
                "SELECT memory_id, links FROM memories WHERE links LIKE ?",
                (f'%"{memory_id}"%',),
            )
            for mem_id, links_json in cursor.fetchall():
                links = _json_loads(links_json)
                if memory_id in links:  # Exact match after parsing
                    links.remove(memory_id)
                    conn.execute(
                        "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                        (_json_dumps(links), now, mem_id),
                    )

            # Delete the memory
            cursor = conn.execute(
                "DELETE FROM memories WHERE memory_id = ?",
                (memory_id,),
            )
            conn.commit()

            if cursor.rowcount == 0:
                return {"deleted": False, "error": f"Memory not found: {memory_id}"}

            return {"deleted": True, "memory_id": memory_id}

    except Exception as e:
        return {"deleted": False, "error": str(e)}


@mcp.tool()
def get_memory_stats() -> dict[str, Any]:
    """
    Get statistics about the memory system.

    Returns:
        Statistics including total memories, storage info, and system status
    """
    try:
        with _get_db() as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            total = cursor.fetchone()[0]

            # Count by project
            cursor = conn.execute(
                "SELECT project, COUNT(*) FROM memories WHERE project IS NOT NULL GROUP BY project"
            )
            projects = dict(cursor.fetchall())

            # Count memories with embeddings
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            )
            with_embeddings = cursor.fetchone()[0]

            # Get top tags
            cursor = conn.execute("SELECT tags FROM memories")
            tags_count: dict[str, int] = {}
            for row in cursor.fetchall():
                for tag in _json_loads(row[0]):
                    if not tag.startswith("project:"):
                        tags_count[tag] = tags_count.get(tag, 0) + 1

            # Check capabilities
            model = _get_embedding_model()
            nlp = _get_nlp()

            return {
                "total_memories": total,
                "with_embeddings": with_embeddings,
                "memories_by_project": projects,
                "top_tags": dict(
                    sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "storage_path": str(DB_PATH),
                "capabilities": {
                    "vector_search": model is not None,
                    "nlp_keywords": nlp is not None,
                },
            }

    except Exception as e:
        return {"error": str(e), "storage_path": str(DB_PATH)}


@mcp.tool()
def evolve_now() -> dict[str, Any]:
    """
    Manually trigger memory evolution/consolidation.

    Re-analyzes all memories to:
    - Improve keyword extraction with latest spaCy
    - Discover new relationships between memories
    - Strengthen link quality

    Returns:
        Consolidation statistics
    """
    try:
        with _get_db() as conn:
            cursor = conn.execute(
                "SELECT memory_id, content, keywords, links FROM memories"
            )
            rows = cursor.fetchall()

            consolidated = 0
            evolved = 0
            total = len(rows)
            now = datetime.now().isoformat()

            for row in rows:
                mem_id, content, old_kw_json, old_links_json = row

                # Progress logging for large datasets
                if consolidated > 0 and consolidated % 100 == 0:
                    logger.info(f"Evolving memories: {consolidated}/{total} processed")

                # Re-extract keywords
                keywords = _extract_keywords(content)
                old_keywords = _json_loads(old_kw_json)

                # Re-find links
                new_links = _find_links(conn, keywords, content, exclude_id=mem_id)
                old_links_set = set(_json_loads(old_links_json))
                new_links_set = set(new_links)

                # Check if evolution needed
                if set(keywords) != set(old_keywords) or new_links_set != old_links_set:
                    # Re-generate embedding
                    embedding = _embed(content)

                    conn.execute(
                        """
                        UPDATE memories SET
                            keywords = ?, links = ?, embedding = ?, updated_at = ?
                        WHERE memory_id = ?
                        """,
                        (
                            _json_dumps(keywords),
                            _json_dumps(new_links),
                            embedding,
                            now,
                            mem_id,
                        ),
                    )

                    # Update bidirectional links
                    added = new_links_set - old_links_set
                    for linked_id in added:
                        link_cursor = conn.execute(
                            "SELECT links FROM memories WHERE memory_id = ?",
                            (linked_id,),
                        )
                        link_row = link_cursor.fetchone()
                        if link_row:
                            existing = _json_loads(link_row[0])
                            if mem_id not in existing:
                                existing.append(mem_id)
                                conn.execute(
                                    "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                                    (_json_dumps(existing[:10]), now, linked_id),
                                )

                    removed = old_links_set - new_links_set
                    for linked_id in removed:
                        link_cursor = conn.execute(
                            "SELECT links FROM memories WHERE memory_id = ?",
                            (linked_id,),
                        )
                        link_row = link_cursor.fetchone()
                        if link_row:
                            existing = _json_loads(link_row[0])
                            if mem_id in existing:
                                existing.remove(mem_id)
                                conn.execute(
                                    "UPDATE memories SET links = ?, updated_at = ? WHERE memory_id = ?",
                                    (_json_dumps(existing), now, linked_id),
                                )

                    evolved += 1

                consolidated += 1

            conn.commit()

            return {
                "status": "completed",
                "memories_processed": consolidated,
                "memories_evolved": evolved,
            }

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def main():
    """Entry point for the A-MEM MCP server."""
    logger.info(f"Starting A-MEM MCP server, database: {DB_PATH}")
    mcp.run()


if __name__ == "__main__":
    main()
