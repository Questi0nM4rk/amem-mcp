"""
A-MEM MCP Server - Brain-like Memory for Claude Code

Based on NeurIPS 2025 paper "A-MEM: Agentic Memory for LLM Agents"
Implements Zettelkasten-inspired memory with dynamic linking and evolution.

Features:
- Persistent ChromaDB storage with semantic search
- Automatic memory linking (new memories connect to related existing ones)
- Memory evolution (new info updates existing memory context)
- Rich metadata generation (keywords, context, tags)
- Global memory shared across all projects
- LLM-enhanced metadata when API key available

Tools:
- store_memory: Store knowledge with automatic linking and evolution
- search_memory: Semantic search across all memories
- read_memory: Read specific memory by ID with full metadata
- list_memories: List recent memories with filtering
- update_memory: Update existing memory (triggers re-evolution)
- delete_memory: Remove a memory
- get_memory_stats: Statistics about the memory system
- evolve_now: Manually trigger memory consolidation and re-linking

Evolution Features (when OPENAI_API_KEY is set):
- LLM-driven evolution analysis for new memories
- Bidirectional link strengthening between related memories
- Neighbor context updates when new knowledge is added
- Automatic consolidation every 10 memories
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent storage directory (global - shared across all projects)
DATA_DIR = Path(os.environ.get("CODEAGENT_HOME", Path.home() / ".codeagent")) / "memory"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastMCP server
mcp = FastMCP(
    "amem",
    instructions="Brain-like memory for Claude Code using A-MEM architecture. "
    "Memories automatically link to each other and evolve over time. "
    "Use this for storing and retrieving project knowledge, patterns, and decisions.",
)

# Global memory system instance (lazy loaded)
_memory_system = None
_use_fallback = False

# spaCy NLP pipeline (lazy loaded singleton)
_nlp = None


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
            # Model not downloaded, try to download it
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


class MemoryNote:
    """Represents a memory note with all metadata."""

    def __init__(
        self,
        id: str,
        content: str,
        keywords: list[str] = None,
        tags: list[str] = None,
        context: str = "",
        links: list[str] = None,
        created_at: str = None,
        updated_at: str = None,
    ):
        self.id = id
        self.content = content
        self.keywords = keywords or []
        self.tags = tags or []
        self.context = context
        self.links = links or []
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at


class PersistentMemorySystem:
    """
    Persistent memory system using ChromaDB directly.

    Fixes issues with the agentic-memory library:
    - Uses chromadb.PersistentClient directly for reliable persistence
    - Always passes embedding_function when getting/creating collections
    - Never calls reset() on existing data
    - Supports LLM-enhanced metadata when API key available
    """

    # Stopwords for keyword extraction
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those', 'it',
        'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'also', 'use',
        'used', 'using', 'uses', 'get', 'gets', 'got', 'make', 'makes', 'made',
    }

    def __init__(
        self,
        directory: str,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize persistent memory system.

        Args:
            directory: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            model_name: SentenceTransformer model for embeddings
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with PersistentClient
        self.client = chromadb.PersistentClient(path=str(self.directory))

        # Create embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # Get or create collection - ALWAYS pass embedding_function
        # This fixes the library bug where extend=True doesn't pass it
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # Persistent counter for memory IDs
        self._counter_file = self.directory / "counter.json"
        self._counter = self._load_counter()

        # Check for LLM availability
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._llm_client = None
        if self._openai_key:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=self._openai_key)
                logger.info("LLM-enhanced metadata enabled (OpenAI)")
            except ImportError:
                logger.info("OpenAI package not installed, using basic metadata")

        logger.info(f"Initialized PersistentMemorySystem at {self.directory}")

        # Evolution tracking
        self._evolution_counter = 0
        self._evolution_threshold = 10  # Evolve every 10 memories
        self._last_evolution = None

        # Evolution system prompt (from original A-MEM paper)
        self._evolution_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note and its nearest neighbors to determine if relationships should evolve.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{neighbors}

Determine:
1. Should this memory form stronger connections with any neighbors?
2. Should any neighbor contexts be updated with new knowledge?

Return JSON:
{{
    "should_evolve": true/false,
    "actions": ["strengthen", "update_neighbor"],
    "strengthen_with": ["memory_ids to link bidirectionally"],
    "neighbor_updates": [
        {{"id": "mem_id", "new_context": "updated context with new knowledge"}}
    ]
}}
'''

    def _load_counter(self) -> int:
        """Load persistent counter for memory IDs."""
        if self._counter_file.exists():
            try:
                with open(self._counter_file, "r") as f:
                    data = json.load(f)
                    return data.get("counter", 0)
            except Exception:
                return 0
        return 0

    def _save_counter(self):
        """Save counter atomically."""
        temp_file = self._counter_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump({"counter": self._counter}, f)
            temp_file.replace(self._counter_file)
        except Exception as e:
            logger.error(f"Failed to save counter: {e}")

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract keywords using spaCy NLP pipeline.
        Falls back to NLTK, then simple tokenization.

        Features with spaCy:
        - Lemmatization (groups word variants)
        - POS filtering (nouns, verbs, adjectives, proper nouns)
        - Named entity recognition (technologies, orgs)
        - Better tokenization
        """
        nlp = _get_nlp()

        if nlp is not None:
            return self._extract_keywords_spacy(text, nlp)
        else:
            return self._extract_keywords_basic(text)

    def _extract_keywords_spacy(self, text: str, nlp) -> list[str]:
        """Extract keywords using spaCy."""
        # Process text (limit to 100k chars for performance)
        doc = nlp(text[:100000])

        keywords = []
        seen_lemmas = set()

        # Extract from tokens with relevant POS
        relevant_pos = {'NOUN', 'VERB', 'ADJ', 'PROPN'}
        for token in doc:
            if (
                token.pos_ in relevant_pos and
                not token.is_stop and
                not token.is_punct and
                len(token.text) > 2 and
                token.is_alpha
            ):
                lemma = token.lemma_.lower()
                if lemma not in seen_lemmas and lemma not in self.STOPWORDS:
                    keywords.append(lemma)
                    seen_lemmas.add(lemma)

        # Add named entities (technologies, organizations, products)
        entity_labels = {'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW'}
        for ent in doc.ents:
            if ent.label_ in entity_labels:
                ent_text = ent.text.lower()
                if ent_text not in seen_lemmas and len(ent_text) > 2:
                    keywords.append(ent_text)
                    seen_lemmas.add(ent_text)

        return keywords[:20]  # Increased cap for richer extraction

    def _extract_keywords_basic(self, text: str) -> list[str]:
        """Fallback: basic keyword extraction without spaCy."""
        # Try NLTK first
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            try:
                words = word_tokenize(text.lower())
                all_stopwords = self.STOPWORDS | set(stopwords.words('english'))
            except LookupError:
                words = text.lower().split()
                all_stopwords = self.STOPWORDS
        except ImportError:
            words = text.lower().split()
            all_stopwords = self.STOPWORDS

        # Filter and clean
        keywords = []
        seen = set()
        for word in words:
            cleaned = word.strip('.,!?;:()[]{}"\'-')
            if (
                len(cleaned) > 2 and
                cleaned not in all_stopwords and
                cleaned.isalnum() and
                cleaned not in seen
            ):
                keywords.append(cleaned)
                seen.add(cleaned)

        return keywords[:15]

    def _generate_context(self, content: str, keywords: list[str]) -> str:
        """
        Generate context for a memory.
        Uses LLM if available, otherwise creates a simple summary.
        """
        if self._llm_client:
            try:
                response = self._llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Generate a one-sentence context/summary for this memory. Be concise."
                        },
                        {
                            "role": "user",
                            "content": f"Content: {content[:500]}\nKeywords: {', '.join(keywords[:10])}"
                        }
                    ],
                    max_tokens=100,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.debug(f"LLM context generation failed: {e}")

        # Fallback: simple context
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        return f"Memory added on {timestamp}. Keywords: {', '.join(keywords[:5])}"

    def _find_links(self, keywords: list[str], content: str, exclude_id: str = None) -> list[str]:
        """
        Find related memories using semantic + lemma-based matching.
        Keywords are already lemmatized if spaCy was used.
        """
        if self.collection.count() == 0:
            return []

        try:
            # Semantic search via ChromaDB embeddings
            results = self.collection.query(
                query_texts=[content],
                n_results=min(10, self.collection.count()),
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            # Convert keywords to lowercase set for matching
            keyword_set = set(kw.lower() for kw in keywords)

            links = []
            for i, doc_id in enumerate(results["ids"][0]):
                if exclude_id and doc_id == exclude_id:
                    continue

                # Get metadata and check keyword overlap
                if i < len(results.get("metadatas", [[]])[0]):
                    metadata = results["metadatas"][0][i]
                    stored_keywords = metadata.get("keywords", "")
                    if isinstance(stored_keywords, str):
                        try:
                            stored_keywords = json.loads(stored_keywords)
                        except (json.JSONDecodeError, TypeError):
                            stored_keywords = []

                    # Match on lemmas (keywords are already lemmatized)
                    stored_set = set(kw.lower() for kw in stored_keywords)
                    overlap = len(keyword_set & stored_set)

                    if overlap >= 2:  # Link threshold
                        links.append(doc_id)

                if len(links) >= 5:
                    break

            return links

        except Exception as e:
            logger.error(f"Error finding links: {e}")
            return []

    def _analyze_evolution(self, memory_id: str, content: str, keywords: list, context: str) -> dict:
        """
        Use LLM to analyze if memory should evolve relationships.
        Returns evolution decisions.
        """
        if not self._llm_client:
            return {"should_evolve": False}

        # Get nearest neighbors
        try:
            results = self.collection.query(
                query_texts=[content],
                n_results=min(5, self.collection.count()),
                include=["documents", "metadatas"],
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return {"should_evolve": False}

            # Format neighbors for prompt
            neighbors_text = ""
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id == memory_id:
                    continue
                if i < len(results.get("documents", [[]])[0]):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i] if i < len(results.get("metadatas", [[]])[0]) else {}
                    neighbors_text += f"- ID: {doc_id}\n  Content: {doc[:200]}...\n  Context: {meta.get('context', '')}\n\n"

            if not neighbors_text:
                return {"should_evolve": False}

            # Call LLM for evolution analysis
            prompt = self._evolution_prompt.format(
                context=context,
                content=content[:500],
                keywords=", ".join(keywords[:10]),
                neighbors=neighbors_text,
            )

            response = self._llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a memory evolution agent. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.warning(f"Evolution analysis failed: {e}")
            return {"should_evolve": False}

    def _apply_evolution(self, memory_id: str, evolution: dict):
        """Apply evolution decisions: strengthen links, update neighbors."""
        if not evolution.get("should_evolve"):
            return

        actions = evolution.get("actions", [])

        # Strengthen bidirectional links
        if "strengthen" in actions:
            strengthen_ids = evolution.get("strengthen_with", [])
            for neighbor_id in strengthen_ids:
                # Add reverse link from neighbor to new memory
                self._add_reverse_link(neighbor_id, memory_id)

        # Update neighbor contexts
        if "update_neighbor" in actions:
            updates = evolution.get("neighbor_updates", [])
            for update in updates:
                neighbor_id = update.get("id")
                new_context = update.get("new_context")
                if neighbor_id and new_context:
                    self._update_neighbor_context(neighbor_id, new_context)

    def _add_reverse_link(self, from_id: str, to_id: str):
        """Add reverse link from existing memory to new memory."""
        try:
            note = self.read(from_id)
            if note:
                links = list(note.links)
                if to_id not in links:
                    links.append(to_id)
                    # Update in ChromaDB
                    result = self.collection.get(ids=[from_id], include=["documents", "metadatas"])
                    if result and result.get("ids"):
                        content = result["documents"][0] if result.get("documents") else ""
                        metadata = result["metadatas"][0] if result.get("metadatas") else {}
                        metadata["links"] = json.dumps(links[:10])  # Cap at 10 links
                        metadata["updated_at"] = datetime.now().isoformat()
                        self.collection.delete(ids=[from_id])
                        self.collection.add(documents=[content], metadatas=[metadata], ids=[from_id])
                        logger.debug(f"Added reverse link {from_id} -> {to_id}")
        except Exception as e:
            logger.warning(f"Failed to add reverse link {from_id} -> {to_id}: {e}")

    def _update_neighbor_context(self, memory_id: str, new_context: str):
        """Update a neighbor's context with new knowledge."""
        try:
            result = self.collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if result and result.get("ids"):
                content = result["documents"][0] if result.get("documents") else ""
                metadata = result["metadatas"][0] if result.get("metadatas") else {}

                # Append new context to existing (evolution history)
                old_context = metadata.get("context", "")
                metadata["context"] = f"{old_context} | Evolved: {new_context[:200]}"
                metadata["updated_at"] = datetime.now().isoformat()

                self.collection.delete(ids=[memory_id])
                self.collection.add(documents=[content], metadatas=[metadata], ids=[memory_id])
                logger.debug(f"Updated neighbor context for {memory_id}")
        except Exception as e:
            logger.warning(f"Failed to update neighbor context {memory_id}: {e}")

    def consolidate_memories(self) -> dict:
        """
        Re-analyze all memories and improve relationships.
        Called periodically or manually via evolve_now() tool.
        """
        if self.collection.count() == 0:
            return {"consolidated": 0, "evolved": 0}

        consolidated = 0
        evolved = 0

        try:
            # Get all memories
            all_results = self.collection.get(
                include=["documents", "metadatas"],
                limit=1000,
            )

            if not all_results or not all_results.get("ids"):
                return {"consolidated": 0, "evolved": 0}

            for i, memory_id in enumerate(all_results["ids"]):
                content = all_results["documents"][i] if i < len(all_results.get("documents", [])) else ""
                metadata = all_results["metadatas"][i] if i < len(all_results.get("metadatas", [])) else {}

                # Re-extract keywords with current spaCy
                keywords = self._extract_keywords(content)

                # Re-find links
                new_links = self._find_links(keywords, content, exclude_id=memory_id)

                # Parse existing data
                old_keywords = metadata.get("keywords", "[]")
                if isinstance(old_keywords, str):
                    try:
                        old_keywords = json.loads(old_keywords)
                    except (json.JSONDecodeError, TypeError):
                        old_keywords = []

                old_links = metadata.get("links", "[]")
                if isinstance(old_links, str):
                    try:
                        old_links = json.loads(old_links)
                    except (json.JSONDecodeError, TypeError):
                        old_links = []

                # Check if evolution needed
                if set(keywords) != set(old_keywords) or set(new_links) != set(old_links):
                    # Update metadata
                    metadata["keywords"] = json.dumps(keywords)
                    metadata["links"] = json.dumps(new_links)
                    metadata["updated_at"] = datetime.now().isoformat()

                    # Update in ChromaDB
                    self.collection.delete(ids=[memory_id])
                    self.collection.add(documents=[content], metadatas=[metadata], ids=[memory_id])

                    evolved += 1

                consolidated += 1

            self._last_evolution = datetime.now().isoformat()
            logger.info(f"Consolidated {consolidated} memories, {evolved} evolved")

            return {"consolidated": consolidated, "evolved": evolved}

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return {"consolidated": consolidated, "evolved": evolved, "error": str(e)}

    def add_note(self, content: str, tags: list = None, **kwargs) -> str:
        """Add a new memory note with evolution."""
        self._counter += 1
        mem_id = f"mem_{self._counter:04d}"
        self._save_counter()

        # Extract metadata
        keywords = self._extract_keywords(content)
        context = self._generate_context(content, keywords)
        links = self._find_links(keywords, content, exclude_id=mem_id)

        # Prepare metadata for ChromaDB (all values must be strings/ints/floats)
        metadata = {
            "id": mem_id,
            "keywords": json.dumps(keywords),
            "tags": json.dumps(tags or []),
            "context": context,
            "links": json.dumps(links),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Store in ChromaDB
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[mem_id],
        )

        # Trigger LLM evolution analysis (if available)
        if self._llm_client:
            evolution = self._analyze_evolution(mem_id, content, keywords, context)
            self._apply_evolution(mem_id, evolution)

        # Check consolidation threshold
        self._evolution_counter += 1
        if self._evolution_counter >= self._evolution_threshold:
            self._evolution_counter = 0
            logger.info("Evolution threshold reached, consolidating...")
            self.consolidate_memories()

        return mem_id

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory by ID."""
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )

            if not result or not result.get("ids") or not result["ids"]:
                return None

            content = result["documents"][0] if result.get("documents") else ""
            metadata = result["metadatas"][0] if result.get("metadatas") else {}

            # Parse JSON fields
            def parse_json_field(field_name, default):
                value = metadata.get(field_name, default)
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return default
                return value if value else default

            return MemoryNote(
                id=memory_id,
                content=content,
                keywords=parse_json_field("keywords", []),
                tags=parse_json_field("tags", []),
                context=metadata.get("context", ""),
                links=parse_json_field("links", []),
                created_at=metadata.get("created_at", ""),
                updated_at=metadata.get("updated_at", ""),
            )

        except Exception as e:
            logger.error(f"Error reading memory {memory_id}: {e}")
            return None

    def search_agentic(self, query: str, k: int = 5) -> list[dict]:
        """Search memories using semantic similarity."""
        if self.collection.count() == 0:
            return []

        try:
            # For empty query, get all memories
            if not query.strip():
                results = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=k,
                )
                ids = results.get("ids", [])
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(k, self.collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
                ids = results.get("ids", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

            memories = []
            for i, doc_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}

                # Parse JSON fields
                def parse_json_field(field_name, default):
                    value = metadata.get(field_name, default)
                    if isinstance(value, str):
                        try:
                            return json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            return default
                    return value if value else default

                # Calculate score from distance (lower distance = higher score)
                if query.strip() and "distances" in results:
                    distances = results.get("distances", [[]])[0]
                    distance = distances[i] if i < len(distances) else 1.0
                    score = 1.0 / (1.0 + distance)  # Convert to similarity score
                else:
                    score = 1.0  # No query, all equally relevant

                memories.append({
                    "id": doc_id,
                    "content": content,
                    "keywords": parse_json_field("keywords", []),
                    "tags": parse_json_field("tags", []),
                    "context": metadata.get("context", ""),
                    "links": parse_json_field("links", []),
                    "score": round(score, 3),
                })

            return memories

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def update(self, memory_id: str, content: str = None, **kwargs) -> bool:
        """Update an existing memory."""
        try:
            # Get existing memory
            existing = self.read(memory_id)
            if not existing:
                return False

            new_content = content if content else existing.content

            # Re-extract metadata if content changed
            if content:
                keywords = self._extract_keywords(new_content)
                context = self._generate_context(new_content, keywords)
                links = self._find_links(keywords, new_content, exclude_id=memory_id)
            else:
                keywords = existing.keywords
                context = existing.context
                links = existing.links

            # Update metadata
            metadata = {
                "id": memory_id,
                "keywords": json.dumps(keywords),
                "tags": json.dumps(existing.tags),
                "context": context,
                "links": json.dumps(links),
                "created_at": existing.created_at,
                "updated_at": datetime.now().isoformat(),
            }

            # Delete and re-add (ChromaDB doesn't have update)
            self.collection.delete(ids=[memory_id])
            self.collection.add(
                documents=[new_content],
                metadatas=[metadata],
                ids=[memory_id],
            )

            return True

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


class _SimpleFallbackMemory:
    """
    Simple fallback memory when ChromaDB is not available.
    Stores memories in JSON with basic keyword extraction.
    """

    def __init__(self):
        self.storage_file = DATA_DIR / "memories.json"
        self.memories = self._load()

    def _load(self) -> dict:
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"memories": {}, "counter": 0}
        return {"memories": {}, "counter": 0}

    def _save(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.memories, f, indent=2)

    def _extract_keywords(self, text: str) -> list[str]:
        """Simple keyword extraction."""
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'but', 'or', 'not', 'this', 'that', 'it', 'its'}
        words = text.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stopwords and len(w) > 2]
        return list(set(keywords))[:10]

    def _find_links(self, keywords: list[str]) -> list[str]:
        """Find related memories based on keyword overlap."""
        links = []
        for mem_id, mem in self.memories.get("memories", {}).items():
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(set(keywords) & mem_keywords)
            if overlap >= 2:  # At least 2 keywords in common
                links.append(mem_id)
        return links[:5]  # Max 5 links

    def add_note(self, content: str, tags: list = None, **kwargs) -> str:
        """Add a new memory."""
        self.memories["counter"] = self.memories.get("counter", 0) + 1
        mem_id = f"mem_{self.memories['counter']:04d}"

        keywords = self._extract_keywords(content)
        links = self._find_links(keywords)

        memory = {
            "id": mem_id,
            "content": content,
            "keywords": keywords,
            "tags": tags or [],
            "context": f"Memory added on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "links": links,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        self.memories.setdefault("memories", {})[mem_id] = memory
        self._save()

        return mem_id

    def read(self, memory_id: str):
        """Read a memory by ID."""
        mem = self.memories.get("memories", {}).get(memory_id)
        if not mem:
            return None

        return MemoryNote(
            id=mem["id"],
            content=mem["content"],
            keywords=mem["keywords"],
            tags=mem["tags"],
            context=mem["context"],
            links=mem.get("links", []),
            created_at=mem["created_at"],
        )

    def search_agentic(self, query: str, k: int = 5) -> list[dict]:
        """Search memories by keyword overlap."""
        query_keywords = set(self._extract_keywords(query))

        scored = []
        for mem_id, mem in self.memories.get("memories", {}).items():
            mem_keywords = set(mem.get("keywords", []))
            mem_content = mem.get("content", "").lower()

            # Score by keyword overlap and content match
            keyword_score = len(query_keywords & mem_keywords) / max(len(query_keywords), 1)
            content_score = sum(1 for kw in query_keywords if kw in mem_content) / max(len(query_keywords), 1)
            score = keyword_score * 0.6 + content_score * 0.4

            if score > 0 or not query.strip():  # Include all if no query
                scored.append((score if query.strip() else 1.0, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem in scored[:k]:
            results.append({
                "id": mem["id"],
                "content": mem["content"],
                "keywords": mem["keywords"],
                "tags": mem["tags"],
                "context": mem.get("context", ""),
                "links": mem.get("links", []),
                "score": round(score, 3),
            })

        return results

    def update(self, memory_id: str, content: str = None, **kwargs) -> bool:
        """Update a memory."""
        if memory_id not in self.memories.get("memories", {}):
            return False

        if content:
            self.memories["memories"][memory_id]["content"] = content
            self.memories["memories"][memory_id]["keywords"] = self._extract_keywords(content)
            self.memories["memories"][memory_id]["links"] = self._find_links(
                self.memories["memories"][memory_id]["keywords"]
            )

        self.memories["memories"][memory_id]["updated_at"] = datetime.now().isoformat()
        self._save()
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self.memories.get("memories", {}):
            del self.memories["memories"][memory_id]
            self._save()
            return True
        return False


def _get_memory_system():
    """Get or initialize the memory system."""
    global _memory_system, _use_fallback

    if _memory_system is not None:
        return _memory_system

    try:
        persist_dir = DATA_DIR / "chromadb"
        persist_dir.mkdir(parents=True, exist_ok=True)

        _memory_system = PersistentMemorySystem(
            directory=str(persist_dir),
            collection_name="memories",
            model_name="all-MiniLM-L6-v2",
        )
        _use_fallback = False
        logger.info(f"Initialized A-MEM with persistent ChromaDB at {persist_dir}")

    except ImportError as e:
        logger.warning(f"ChromaDB not available, using fallback: {e}")
        _use_fallback = True
        _memory_system = _SimpleFallbackMemory()

    except Exception as e:
        logger.warning(f"Failed to initialize ChromaDB, using fallback: {e}")
        _use_fallback = True
        _memory_system = _SimpleFallbackMemory()

    return _memory_system


def _format_note(note) -> dict:
    """Format a memory note for response."""
    if isinstance(note, MemoryNote):
        return {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "context": note.context,
            "links": note.links,
        }
    elif isinstance(note, dict):
        return {
            "id": note.get("id", ""),
            "content": note.get("content", ""),
            "keywords": note.get("keywords", []),
            "tags": note.get("tags", []),
            "context": note.get("context", ""),
            "links": note.get("links", []),
        }
    return {}


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
    memory = _get_memory_system()

    # Add project as tag if provided
    all_tags = list(tags) if tags else []
    if project:
        all_tags.append(f"project:{project}")

    try:
        memory_id = memory.add_note(content=content, tags=all_tags)

        # Read back to get generated metadata
        note = memory.read(memory_id)

        if note:
            formatted = _format_note(note)
            return {
                "stored": True,
                "memory_id": memory_id,
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "keywords": formatted.get("keywords", []),
                "context": formatted.get("context", ""),
                "linked_to": len(formatted.get("links", [])),
                "using_fallback": _use_fallback,
                "storage_path": str(DATA_DIR),
            }
        else:
            return {
                "stored": True,
                "memory_id": memory_id,
                "content_preview": content[:100],
                "using_fallback": _use_fallback,
            }

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        return {
            "stored": False,
            "error": str(e),
        }


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
    memory = _get_memory_system()
    k = min(max(k, 1), 20)

    try:
        results = memory.search_agentic(query, k=k)

        # Filter by project if specified
        if project:
            project_tag = f"project:{project}"
            results = [r for r in results if project_tag in r.get("tags", [])]

        formatted = []
        for result in results:
            formatted.append({
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "keywords": result.get("keywords", []),
                "tags": result.get("tags", []),
                "context": result.get("context", ""),
                "links": result.get("links", []),
                "relevance": result.get("score", 0),
            })

        return {
            "query": query,
            "results": formatted,
            "count": len(formatted),
            "using_fallback": _use_fallback,
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": str(e),
        }


@mcp.tool()
def read_memory(memory_id: str) -> dict[str, Any]:
    """
    Read a specific memory by ID with full metadata.

    Args:
        memory_id: The memory ID to retrieve

    Returns:
        Full memory content with keywords, context, tags, and links
    """
    memory = _get_memory_system()

    try:
        note = memory.read(memory_id)

        if note:
            formatted = _format_note(note)
            formatted["found"] = True
            return formatted
        else:
            return {
                "found": False,
                "error": f"Memory not found: {memory_id}",
            }

    except Exception as e:
        return {
            "found": False,
            "error": str(e),
        }


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
    memory = _get_memory_system()
    limit = min(max(limit, 1), 50)

    try:
        # Use search with empty query to get all, then filter
        results = memory.search_agentic("", k=limit * 2)  # Get more to account for filtering

        # Filter
        if project:
            project_tag = f"project:{project}"
            results = [r for r in results if project_tag in r.get("tags", [])]

        if tag:
            results = [r for r in results if tag in r.get("tags", [])]

        results = results[:limit]

        return {
            "memories": [{
                "id": r.get("id", ""),
                "content_preview": r.get("content", "")[:100],
                "keywords": r.get("keywords", [])[:5],
                "tags": r.get("tags", []),
            } for r in results],
            "count": len(results),
            "filters": {
                "project": project,
                "tag": tag,
            },
        }

    except Exception as e:
        return {
            "memories": [],
            "count": 0,
            "error": str(e),
        }


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
    memory = _get_memory_system()

    try:
        result = memory.update(memory_id, content=content)

        if result:
            note = memory.read(memory_id)
            if note:
                formatted = _format_note(note)
                return {
                    "updated": True,
                    "memory_id": memory_id,
                    "new_keywords": formatted.get("keywords", []),
                    "new_links": len(formatted.get("links", [])),
                }

        return {
            "updated": False,
            "error": f"Memory not found: {memory_id}",
        }

    except Exception as e:
        return {
            "updated": False,
            "error": str(e),
        }


@mcp.tool()
def delete_memory(memory_id: str) -> dict[str, Any]:
    """
    Delete a memory.

    Args:
        memory_id: The memory to delete

    Returns:
        Confirmation of deletion
    """
    memory = _get_memory_system()

    try:
        result = memory.delete(memory_id)
        return {
            "deleted": bool(result),
            "memory_id": memory_id,
        }
    except Exception as e:
        return {
            "deleted": False,
            "error": str(e),
        }


@mcp.tool()
def get_memory_stats() -> dict[str, Any]:
    """
    Get statistics about the memory system.

    Returns:
        Statistics including total memories, storage info, and system status
    """
    memory = _get_memory_system()

    try:
        # Count memories by doing a broad search
        all_results = memory.search_agentic("", k=1000)
        total = len(all_results)

        # Count by project
        projects = {}
        tags_count = {}
        for r in all_results:
            for tag in r.get("tags", []):
                if tag.startswith("project:"):
                    proj = tag[8:]
                    projects[proj] = projects.get(proj, 0) + 1
                else:
                    tags_count[tag] = tags_count.get(tag, 0) + 1

        # Determine backend info
        if _use_fallback:
            backend = "JSON (fallback)"
        else:
            backend = "ChromaDB (persistent)"
            # Check if LLM enhanced
            if hasattr(memory, '_llm_client') and memory._llm_client:
                backend += " + LLM metadata"

        return {
            "total_memories": total,
            "memories_by_project": projects,
            "top_tags": dict(sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]),
            "storage_path": str(DATA_DIR),
            "using_fallback": _use_fallback,
            "backend": backend,
        }

    except Exception as e:
        return {
            "error": str(e),
            "storage_path": str(DATA_DIR),
            "using_fallback": _use_fallback,
        }


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
    memory = _get_memory_system()

    if hasattr(memory, 'consolidate_memories'):
        result = memory.consolidate_memories()
        return {
            "status": "completed",
            "memories_processed": result.get("consolidated", 0),
            "memories_evolved": result.get("evolved", 0),
            "using_fallback": _use_fallback,
        }
    else:
        return {
            "status": "not_available",
            "reason": "Evolution not supported in fallback mode",
        }


def main():
    """Entry point for the A-MEM MCP server."""
    logger.info(f"Starting A-MEM MCP server, storage: {DATA_DIR}")
    mcp.run()


if __name__ == "__main__":
    main()
