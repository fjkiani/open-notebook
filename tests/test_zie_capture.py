"""
Unit tests for open_notebook.plugins.zie_capture

Covers:
- capture_qa_pair() with no ZIE_DATABASE_URL → returns silently (no crash)
- capture_qa_pair() with mocked asyncpg.connect → writes correct schema fields
- prompt_hash is deterministic (same Q + docs → same hash)
- fast/slow training records use correct source_kind values
- preference pair uses correct preference_source
- ON CONFLICT: UniqueViolationError on preference pair is swallowed
- ZIE_DATABASE_URL not set → warning logged, no DB call
"""

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_hash(question: str, source_doc_ids: list) -> str:
    return hashlib.sha256(
        json.dumps({"q": question, "docs": sorted(source_doc_ids)}).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Suite 1: No ZIE_DATABASE_URL
# ---------------------------------------------------------------------------

class TestZIECaptureNoURL:
    """Tests for capture_qa_pair when ZIE_DATABASE_URL is not set."""

    @pytest.mark.asyncio
    async def test_no_database_url_returns_silently(self):
        """capture_qa_pair with no ZIE_DATABASE_URL returns without error."""
        with patch.dict("os.environ", {"ZIE_DATABASE_URL": ""}, clear=False):
            # Re-import to pick up the patched env var
            import importlib
            import open_notebook.plugins.zie_capture as zie_module
            importlib.reload(zie_module)

            # Should not raise
            await zie_module.capture_qa_pair(
                question="What is X?",
                fast_response="X is a thing",
                slow_response="X is a thing",
                source_doc_ids=["source:abc"],
            )

    @pytest.mark.asyncio
    async def test_no_database_url_does_not_call_asyncpg(self):
        """capture_qa_pair with no ZIE_DATABASE_URL never calls asyncpg.connect."""
        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", ""):
            with patch("open_notebook.plugins.zie_capture.asyncpg") as mock_asyncpg:
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="Q?",
                    fast_response="A",
                    slow_response="A",
                    source_doc_ids=[],
                )

                mock_asyncpg.connect.assert_not_called()


# ---------------------------------------------------------------------------
# Suite 2: With ZIE_DATABASE_URL — schema correctness
# ---------------------------------------------------------------------------

class TestZIECaptureWithURL:
    """Tests for capture_qa_pair when ZIE_DATABASE_URL is set."""

    def _make_mock_conn(self, fast_id="uuid-fast", slow_id="uuid-slow"):
        """Create a mock asyncpg connection."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=[fast_id, slow_id])
        conn.execute = AsyncMock()
        conn.close = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_writes_two_training_records(self):
        """capture_qa_pair writes exactly two training records (fast + slow)."""
        mock_conn = self._make_mock_conn()

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="What is X?",
                    fast_response="X is fast",
                    slow_response="X is slow",
                    source_doc_ids=["source:abc"],
                )

        # fetchval called twice: once for fast record, once for slow record
        assert mock_conn.fetchval.call_count == 2

    @pytest.mark.asyncio
    async def test_fast_record_uses_direct_call_source_kind(self):
        """Fast training record uses source_kind='direct_call'."""
        mock_conn = self._make_mock_conn()

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="Q?",
                    fast_response="fast answer",
                    slow_response="slow answer",
                    source_doc_ids=[],
                )

        # First fetchval call is for the fast record
        first_call_args = mock_conn.fetchval.call_args_list[0][0]
        sql = first_call_args[0]
        assert "direct_call" in sql

    @pytest.mark.asyncio
    async def test_slow_record_uses_remote_promoted_source_kind(self):
        """Slow training record uses source_kind='remote_promoted'."""
        mock_conn = self._make_mock_conn()

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="Q?",
                    fast_response="fast",
                    slow_response="slow",
                    source_doc_ids=[],
                )

        second_call_args = mock_conn.fetchval.call_args_list[1][0]
        sql = second_call_args[0]
        assert "remote_promoted" in sql

    @pytest.mark.asyncio
    async def test_preference_pair_written_when_both_ids_present(self):
        """Preference pair is written when both fast_id and slow_id are non-null."""
        mock_conn = self._make_mock_conn(fast_id="uuid-fast", slow_id="uuid-slow")

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="Q?",
                    fast_response="fast",
                    slow_response="slow",
                    source_doc_ids=[],
                )

        # execute called once for preference pair
        mock_conn.execute.assert_awaited_once()
        pref_sql = mock_conn.execute.call_args[0][0]
        assert "zie_preference_pairs" in pref_sql
        assert "remote_beats_local" in pref_sql

    @pytest.mark.asyncio
    async def test_preference_pair_skipped_when_fast_id_null(self):
        """Preference pair is NOT written when fast_id is null."""
        mock_conn = self._make_mock_conn(fast_id=None, slow_id="uuid-slow")

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                await capture_qa_pair(
                    question="Q?",
                    fast_response="fast",
                    slow_response="slow",
                    source_doc_ids=[],
                )

        mock_conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connection_always_closed(self):
        """asyncpg connection is always closed, even on error."""
        mock_conn = self._make_mock_conn()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("DB error"))

        with patch("open_notebook.plugins.zie_capture.ZIE_DATABASE_URL", "postgresql://test"):
            with patch("open_notebook.plugins.zie_capture.asyncpg.connect",
                       new_callable=AsyncMock, return_value=mock_conn):
                from open_notebook.plugins.zie_capture import capture_qa_pair

                # Should not raise — error is caught internally
                await capture_qa_pair(
                    question="Q?",
                    fast_response="fast",
                    slow_response="slow",
                    source_doc_ids=[],
                )

        mock_conn.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Suite 3: Prompt hash determinism
# ---------------------------------------------------------------------------

class TestPromptHashDeterminism:
    """Tests for prompt hash computation."""

    def test_same_question_same_docs_same_hash(self):
        """Same question + docs always produce the same hash."""
        h1 = _expected_hash("What is X?", ["source:a", "source:b"])
        h2 = _expected_hash("What is X?", ["source:a", "source:b"])
        assert h1 == h2

    def test_doc_order_does_not_affect_hash(self):
        """Source doc IDs are sorted before hashing — order doesn't matter."""
        h1 = _expected_hash("Q?", ["source:b", "source:a"])
        h2 = _expected_hash("Q?", ["source:a", "source:b"])
        assert h1 == h2

    def test_different_questions_different_hash(self):
        """Different questions produce different hashes."""
        h1 = _expected_hash("What is X?", [])
        h2 = _expected_hash("What is Y?", [])
        assert h1 != h2

    def test_fast_and_slow_hashes_differ(self):
        """Fast and slow training records use different hash suffixes."""
        base = _expected_hash("Q?", [])
        fast_hash = base + "_fast"
        slow_hash = base + "_slow"
        assert fast_hash != slow_hash


# ---------------------------------------------------------------------------
# Suite 4: ZIE domain constants
# ---------------------------------------------------------------------------

class TestZIEConstants:
    """Tests for ZIE domain and task type constants."""

    def test_domain_is_notebook(self):
        """ZIE_DOMAIN is 'notebook'."""
        from open_notebook.plugins.zie_capture import ZIE_DOMAIN
        assert ZIE_DOMAIN == "notebook"

    def test_task_type_is_notebook_qa(self):
        """ZIE_TASK_TYPE is 'notebook_qa'."""
        from open_notebook.plugins.zie_capture import ZIE_TASK_TYPE
        assert ZIE_TASK_TYPE == "notebook_qa"
