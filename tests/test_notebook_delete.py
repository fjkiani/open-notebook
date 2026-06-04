"""
Unit tests for Notebook.delete() cascade and get_delete_preview().

Covers:
- get_delete_preview(): returns correct note/exclusive/shared source counts
- delete(delete_exclusive_sources=False): deletes notes, unlinks sources, deletes notebook
- delete(delete_exclusive_sources=True): deletes exclusive sources, unlinks shared sources
- delete() with no ID raises InvalidInputError
- Empty notebook: delete returns all-zero counts
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from open_notebook.domain.notebook import Notebook
from open_notebook.exceptions import InvalidInputError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_notebook(notebook_id="notebook:test123"):
    nb = Notebook(name="Test Notebook", description="desc")
    nb.id = notebook_id
    return nb


def _make_note(note_id="note:abc"):
    note = MagicMock()
    note.id = note_id
    note.delete = AsyncMock()
    return note


def _make_source(source_id="source:abc"):
    source = MagicMock()
    source.id = source_id
    source.delete = AsyncMock()
    return source


# ---------------------------------------------------------------------------
# Suite 1: get_delete_preview
# ---------------------------------------------------------------------------

class TestGetDeletePreview:
    """Tests for Notebook.get_delete_preview()."""

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_returns_correct_counts(self, mock_query):
        """get_delete_preview returns note_count, exclusive_source_count, shared_source_count."""
        nb = _make_notebook()

        # First call: note count query
        # Second call: source counts query
        mock_query.side_effect = [
            [{"count": 3}],  # note_count = 3
            [
                {"id": "source:a", "assigned_others": 0},  # exclusive
                {"id": "source:b", "assigned_others": 0},  # exclusive
                {"id": "source:c", "assigned_others": 2},  # shared
            ],
        ]

        result = await nb.get_delete_preview()

        assert result["note_count"] == 3
        assert result["exclusive_source_count"] == 2
        assert result["shared_source_count"] == 1

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_empty_notebook_returns_zeros(self, mock_query):
        """Empty notebook returns all-zero counts."""
        nb = _make_notebook()

        mock_query.side_effect = [
            [],   # no notes
            [],   # no sources
        ]

        result = await nb.get_delete_preview()

        assert result["note_count"] == 0
        assert result["exclusive_source_count"] == 0
        assert result["shared_source_count"] == 0

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_all_shared_sources(self, mock_query):
        """All sources shared → exclusive_source_count=0, shared_source_count=N."""
        nb = _make_notebook()

        mock_query.side_effect = [
            [{"count": 1}],
            [
                {"id": "source:a", "assigned_others": 3},
                {"id": "source:b", "assigned_others": 1},
            ],
        ]

        result = await nb.get_delete_preview()

        assert result["exclusive_source_count"] == 0
        assert result["shared_source_count"] == 2

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_all_exclusive_sources(self, mock_query):
        """All sources exclusive → shared_source_count=0."""
        nb = _make_notebook()

        mock_query.side_effect = [
            [{"count": 0}],
            [
                {"id": "source:a", "assigned_others": 0},
                {"id": "source:b", "assigned_others": 0},
            ],
        ]

        result = await nb.get_delete_preview()

        assert result["exclusive_source_count"] == 2
        assert result["shared_source_count"] == 0


# ---------------------------------------------------------------------------
# Suite 2: delete(delete_exclusive_sources=False)
# ---------------------------------------------------------------------------

class TestNotebookDeleteKeepSources:
    """Tests for Notebook.delete(delete_exclusive_sources=False)."""

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.ObjectModel.delete", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_deletes_notes_and_unlinks_sources(self, mock_query, mock_super_delete):
        """delete(False) deletes notes, unlinks sources, deletes notebook record."""
        nb = _make_notebook()
        notes = [_make_note("note:1"), _make_note("note:2")]

        # repo_query calls: artifact delete, source count, reference delete
        mock_query.side_effect = [
            None,              # DELETE artifact
            [{"count": 2}],    # SELECT count from reference (unlinked_sources)
            None,              # DELETE reference
        ]

        with patch.object(Notebook, "get_notes", new_callable=AsyncMock, return_value=notes):
            result = await nb.delete(delete_exclusive_sources=False)

        assert result["deleted_notes"] == 2
        assert result["deleted_sources"] == 0
        assert result["unlinked_sources"] == 2

        # Each note.delete() was called
        for note in notes:
            note.delete.assert_awaited_once()

        # Notebook itself was deleted
        mock_super_delete.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.ObjectModel.delete", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_empty_notebook_delete_returns_zeros(self, mock_query, mock_super_delete):
        """Empty notebook delete returns all-zero counts."""
        nb = _make_notebook()

        mock_query.side_effect = [
            None,           # DELETE artifact
            [],             # SELECT count from reference → 0
            None,           # DELETE reference
        ]

        with patch.object(Notebook, "get_notes", new_callable=AsyncMock, return_value=[]):
            result = await nb.delete(delete_exclusive_sources=False)

        assert result["deleted_notes"] == 0
        assert result["deleted_sources"] == 0
        assert result["unlinked_sources"] == 0

    @pytest.mark.asyncio
    async def test_delete_without_id_raises_invalid_input_error(self):
        """delete() on notebook without ID raises InvalidInputError."""
        nb = Notebook(name="No ID", description="")
        nb.id = None

        with pytest.raises(InvalidInputError):
            await nb.delete()


# ---------------------------------------------------------------------------
# Suite 3: delete(delete_exclusive_sources=True)
# ---------------------------------------------------------------------------

class TestNotebookDeleteWithSources:
    """Tests for Notebook.delete(delete_exclusive_sources=True)."""

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.ObjectModel.delete", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.Source.get", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_deletes_exclusive_sources_unlinks_shared(
        self, mock_query, mock_source_get, mock_super_delete
    ):
        """delete(True) deletes exclusive sources, unlinks shared sources."""
        nb = _make_notebook()

        exclusive_source = _make_source("source:exclusive")
        mock_source_get.return_value = exclusive_source

        mock_query.side_effect = [
            None,  # DELETE artifact
            # source_counts query: one exclusive, one shared
            [
                {"id": "source:exclusive", "assigned_others": 0},
                {"id": "source:shared", "assigned_others": 1},
            ],
            None,  # DELETE reference
        ]

        with patch.object(Notebook, "get_notes", new_callable=AsyncMock, return_value=[]):
            result = await nb.delete(delete_exclusive_sources=True)

        assert result["deleted_sources"] == 1
        assert result["unlinked_sources"] == 1
        exclusive_source.delete.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.ObjectModel.delete", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.Source.get", new_callable=AsyncMock)
    @patch("open_notebook.domain.notebook.repo_query", new_callable=AsyncMock)
    async def test_source_delete_failure_is_logged_not_raised(
        self, mock_query, mock_source_get, mock_super_delete
    ):
        """Failure to delete an exclusive source is logged but does not abort the delete."""
        nb = _make_notebook()

        failing_source = _make_source("source:failing")
        failing_source.delete = AsyncMock(side_effect=Exception("disk error"))
        mock_source_get.return_value = failing_source

        mock_query.side_effect = [
            None,
            [{"id": "source:failing", "assigned_others": 0}],
            None,
        ]

        with patch.object(Notebook, "get_notes", new_callable=AsyncMock, return_value=[]):
            # Should not raise — failure is caught and logged
            result = await nb.delete(delete_exclusive_sources=True)

        # deleted_sources stays 0 because delete failed
        assert result["deleted_sources"] == 0
        mock_super_delete.assert_awaited_once()
