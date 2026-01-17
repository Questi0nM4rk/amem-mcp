"""Tests for amem_mcp.server module."""

import pytest

from amem_mcp.server import (
    _extract_keywords_basic,
    _generate_context,
    _json_dumps,
    _json_loads,
)


class TestJsonHelpers:
    """Tests for JSON helper functions."""

    def test_json_loads_valid_list(self) -> None:
        """_json_loads should parse valid JSON array."""
        result = _json_loads('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_loads_none(self) -> None:
        """_json_loads should return empty list for None."""
        result = _json_loads(None)
        assert result == []

    def test_json_loads_invalid(self) -> None:
        """_json_loads should return empty list for invalid JSON."""
        result = _json_loads("not json")
        assert result == []

    def test_json_dumps_list(self) -> None:
        """_json_dumps should serialize list to JSON."""
        result = _json_dumps(["a", "b", "c"])
        assert result == '["a", "b", "c"]'

    def test_json_dumps_none(self) -> None:
        """_json_dumps should return '[]' for None."""
        result = _json_dumps(None)
        assert result == "[]"


class TestKeywordExtraction:
    """Tests for keyword extraction functions."""

    def test_extract_keywords_basic_simple(self) -> None:
        """_extract_keywords_basic should extract words."""
        result = _extract_keywords_basic("hello world test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_extract_keywords_basic_filters_short_words(self) -> None:
        """_extract_keywords_basic should filter short words."""
        result = _extract_keywords_basic("a the is at to")
        # Short words should be filtered
        assert len(result) == 0 or all(len(w) > 2 for w in result)

    def test_extract_keywords_basic_lowercase(self) -> None:
        """_extract_keywords_basic should lowercase keywords."""
        result = _extract_keywords_basic("Hello WORLD Test")
        assert all(w.islower() for w in result)


class TestContextGeneration:
    """Tests for context generation."""

    def test_generate_context_empty(self) -> None:
        """_generate_context should handle empty keywords."""
        result = _generate_context([])
        assert isinstance(result, str)

    def test_generate_context_with_keywords(self) -> None:
        """_generate_context should generate context from keywords."""
        result = _generate_context(["python", "testing", "code"])
        assert isinstance(result, str)
        assert len(result) > 0
