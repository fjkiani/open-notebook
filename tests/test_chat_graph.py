"""
Unit tests for open_notebook.graphs.chat

Covers:
- call_model_with_messages: thinking-tag stripping, model override, error propagation
- Async bridge: ThreadPoolExecutor path (running loop) and asyncio.run() path (no loop)
- Graph compilation: checkpointer present, invoke/ainvoke callable
"""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _make_state(messages=None, model_override=None, context=None):
    return {
        "messages": messages or [HumanMessage(content="Hello")],
        "notebook": None,
        "context": context,
        "context_config": None,
        "model_override": model_override,
    }


# ---------------------------------------------------------------------------
# Suite 1: call_model_with_messages — thinking content cleaning
# ---------------------------------------------------------------------------

class TestCallModelWithMessages:
    """Tests for the sync call_model_with_messages node in chat.py."""

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_plain_response_returned_unchanged(self, mock_run, mock_loop):
        """Normal AI response (no think tags) passes through unchanged."""
        from open_notebook.graphs.chat import call_model_with_messages

        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message("Hello back!")
        mock_run.return_value = mock_model

        result = call_model_with_messages(
            _make_state(), {"configurable": {"model_id": None}}
        )

        assert result["messages"].content == "Hello back!"

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_thinking_tags_stripped(self, mock_run, mock_loop):
        """<think>...</think> blocks are removed from AI response."""
        from open_notebook.graphs.chat import call_model_with_messages

        raw = "<think>internal reasoning here</think>The actual answer."
        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message(raw)
        mock_run.return_value = mock_model

        result = call_model_with_messages(
            _make_state(), {"configurable": {"model_id": None}}
        )

        assert "<think>" not in result["messages"].content
        assert "The actual answer." in result["messages"].content

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_multiline_thinking_tags_stripped(self, mock_run, mock_loop):
        """Multi-line <think> blocks are fully removed."""
        from open_notebook.graphs.chat import call_model_with_messages

        raw = "<think>\nline1\nline2\n</think>\nFinal response."
        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message(raw)
        mock_run.return_value = mock_model

        result = call_model_with_messages(
            _make_state(), {"configurable": {"model_id": None}}
        )

        content = result["messages"].content
        assert "<think>" not in content
        assert "line1" not in content
        assert "Final response." in content

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_open_notebook_error_propagates_unchanged(self, mock_run, mock_loop):
        """OpenNotebookError is re-raised without wrapping."""
        from open_notebook.graphs.chat import call_model_with_messages
        from open_notebook.exceptions import OpenNotebookError

        mock_model = MagicMock()
        mock_model.invoke.side_effect = OpenNotebookError("auth failed")
        mock_run.return_value = mock_model

        with pytest.raises(OpenNotebookError, match="auth failed"):
            call_model_with_messages(
                _make_state(), {"configurable": {"model_id": None}}
            )

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_generic_exception_classified_to_open_notebook_error(self, mock_run, mock_loop):
        """Generic exceptions are classified into OpenNotebookError subclasses."""
        from open_notebook.graphs.chat import call_model_with_messages
        from open_notebook.exceptions import OpenNotebookError

        mock_model = MagicMock()
        mock_model.invoke.side_effect = ValueError("unexpected failure")
        mock_run.return_value = mock_model

        with pytest.raises(OpenNotebookError):
            call_model_with_messages(
                _make_state(), {"configurable": {"model_id": None}}
            )

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_empty_thinking_block_stripped(self, mock_run, mock_loop):
        """Empty <think></think> block is stripped cleanly."""
        from open_notebook.graphs.chat import call_model_with_messages

        raw = "<think></think>Clean answer."
        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message(raw)
        mock_run.return_value = mock_model

        result = call_model_with_messages(
            _make_state(), {"configurable": {"model_id": None}}
        )

        assert "<think>" not in result["messages"].content
        assert "Clean answer." in result["messages"].content


# ---------------------------------------------------------------------------
# Suite 2: Async bridge — ThreadPoolExecutor path
# ---------------------------------------------------------------------------

class TestAsyncBridge:
    """Tests for the sync/async bridge in call_model_with_messages."""

    def test_thread_pool_path_when_loop_running(self):
        """When a running event loop exists, ThreadPoolExecutor path is used."""
        from open_notebook.graphs.chat import call_model_with_messages

        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message("thread pool response")

        with patch("open_notebook.graphs.chat.asyncio.get_running_loop",
                   return_value=MagicMock()):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.__enter__ = MagicMock(return_value=mock_executor)
                mock_executor.__exit__ = MagicMock(return_value=False)
                mock_future = MagicMock()
                mock_future.result.return_value = mock_model
                mock_executor.submit.return_value = mock_future
                mock_executor_cls.return_value = mock_executor

                result = call_model_with_messages(
                    _make_state(), {"configurable": {"model_id": None}}
                )

        assert result["messages"].content == "thread pool response"

    @patch("open_notebook.graphs.chat.asyncio.get_running_loop",
           side_effect=RuntimeError("no loop"))
    @patch("open_notebook.graphs.chat.asyncio.run")
    def test_asyncio_run_path_when_no_loop(self, mock_run, mock_loop):
        """When no event loop is running, asyncio.run() path is used."""
        from open_notebook.graphs.chat import call_model_with_messages

        mock_model = MagicMock()
        mock_model.invoke.return_value = _make_ai_message("asyncio run response")
        mock_run.return_value = mock_model

        result = call_model_with_messages(
            _make_state(), {"configurable": {"model_id": None}}
        )

        assert result["messages"].content == "asyncio run response"
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Suite 3: Graph compilation
# ---------------------------------------------------------------------------

class TestChatGraphCompilation:
    """Tests for the compiled chat graph structure."""

    def test_graph_is_compiled(self):
        """chat.graph is a compiled LangGraph StateGraph."""
        from open_notebook.graphs.chat import graph
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")

    def test_graph_has_sqlite_checkpointer(self):
        """chat.graph uses SqliteSaver as checkpointer."""
        from open_notebook.graphs.chat import memory
        from langgraph.checkpoint.sqlite import SqliteSaver
        assert isinstance(memory, SqliteSaver)

    def test_graph_checkpointer_backed_by_sqlite_connection(self):
        """The graph's checkpointer is backed by a real SQLite connection."""
        from open_notebook.graphs.chat import conn
        assert isinstance(conn, sqlite3.Connection)

    def test_graph_has_agent_node(self):
        """Graph has the 'agent' node."""
        from open_notebook.graphs.chat import graph
        graph_repr = graph.get_graph()
        node_names = list(graph_repr.nodes.keys())
        assert "agent" in node_names

    def test_graph_invoke_is_callable(self):
        """graph.invoke and graph.ainvoke are callable."""
        from open_notebook.graphs.chat import graph
        assert callable(graph.invoke)
        assert callable(graph.ainvoke)
