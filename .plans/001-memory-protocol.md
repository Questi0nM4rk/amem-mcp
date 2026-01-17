# Plan: Define MemoryProtocol for Full/Fallback Modes

From review session (Jan 2026).

## Problem

Full mode (ChromaDB) and Fallback mode (JSON) have different implementations with no shared interface.

## Solution

Define a Protocol (PEP 544) that both modes implement.

```python
@runtime_checkable
class MemoryProtocol(Protocol):
    def add_note(self, content: str, tags: list[str] | None = None) -> str: ...
    def search(self, query: str, k: int = 5) -> list[dict]: ...
    def get(self, memory_id: str) -> dict | None: ...
    def update(self, memory_id: str, content: str) -> bool: ...
    def delete(self, memory_id: str) -> bool: ...
```

## Done Criteria

- [ ] MemoryProtocol defined
- [ ] Both backends implement protocol
- [ ] Factory function selects backend
- [ ] Tests verify conformance
