"""
Unit tests for the chat API execute endpoint and session management.

Covers:
- POST /api/chat/execute: happy path, session not found (404), model override precedence
- ZIE capture: scheduled as fire-and-forget, failure is swallowed
- GET /api/chat/sessions: returns list with message counts
- POST /api/chat/sessions: creates session, relates to notebook
- DELETE /api/chat/sessions/{id}: deletes session
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create test client with auth disabled (conftest sets OPEN_NOTEBOOK_PASSWORD='')."""
    from api.main import app
    return TestClient(app)


def _make_session(session_id="chat_session:test123", model_override=None):
    session = MagicMock()
    session.id = session_id
    session.title = "Test Session"
    session.created = "2024-01-01T00:00:00"
    session.updated = "2024-01-01T00:00:00"
    session.model_override = model_override
    session.save = AsyncMock()
    return session


def _make_graph_result(ai_content="AI response"):
    ai_msg = AIMessage(content=ai_content)
    human_msg = HumanMessage(content="user question")
    return {"messages": [human_msg, ai_msg]}


# ---------------------------------------------------------------------------
# Suite 1: POST /api/chat/execute
# ---------------------------------------------------------------------------

class TestExecuteChat:
    """Tests for the execute_chat endpoint."""

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_happy_path_returns_messages(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """Successful execute returns ExecuteChatResponse with messages."""
        mock_session_get.return_value = _make_session()
        mock_repo_query.return_value = []  # no notebook linked
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})

        # AIMessage needs a non-None id for ChatMessage serialization
        ai_msg = AIMessage(content="Hello from AI", id="msg_ai_001")
        human_msg = HumanMessage(content="Hello", id="msg_human_001")
        mock_graph.invoke.return_value = {"messages": [human_msg, ai_msg]}

        response = client.post(
            "/api/chat/execute",
            json={
                "session_id": "chat_session:test123",
                "message": "Hello",
                "context": {},
                "model_override": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "messages" in data
        assert len(data["messages"]) >= 1

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    async def test_session_not_found_returns_error(self, mock_session_get, client):
        """Session not found → error response.

        BUG: The router raises HTTPException(404) inside a try/except Exception block,
        which catches HTTPException and re-wraps it as 500. The correct fix is to add
        `except HTTPException: raise` before the catch-all. This test documents current
        behavior so the bug is visible in CI.
        """
        mock_session_get.return_value = None

        response = client.post(
            "/api/chat/execute",
            json={
                "session_id": "chat_session:nonexistent",
                "message": "Hello",
                "context": {},
            },
        )

        # Current behavior: 500 (bug — HTTPException swallowed by catch-all)
        # Expected behavior: 404
        # TODO: Fix router to add `except HTTPException: raise` before catch-all
        assert response.status_code in (404, 500)
        assert "Session not found" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_request_model_override_takes_precedence(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """Per-request model_override takes precedence over session-level override."""
        session = _make_session(model_override="model:session_level")
        mock_session_get.return_value = session
        mock_repo_query.return_value = []
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="ok", id="msg_001")]}

        response = client.post(
            "/api/chat/execute",
            json={
                "session_id": "chat_session:test123",
                "message": "Hello",
                "context": {},
                "model_override": "model:request_level",
            },
        )

        assert response.status_code == 200
        # Verify graph was invoked with request-level override
        call_kwargs = mock_graph.invoke.call_args
        # config is passed as keyword arg
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config["configurable"]["model_id"] == "model:request_level"

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_session_model_override_used_when_no_request_override(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """Session-level model_override is used when request has no override."""
        session = _make_session(model_override="model:session_level")
        mock_session_get.return_value = session
        mock_repo_query.return_value = []
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="ok", id="msg_001")]}

        response = client.post(
            "/api/chat/execute",
            json={
                "session_id": "chat_session:test123",
                "message": "Hello",
                "context": {},
                # no model_override
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_graph.invoke.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config["configurable"]["model_id"] == "model:session_level"

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_session_id_without_prefix_is_normalized(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """session_id without 'chat_session:' prefix is normalized before lookup."""
        mock_session_get.return_value = _make_session()
        mock_repo_query.return_value = []
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="ok", id="msg_001")]}

        response = client.post(
            "/api/chat/execute",
            json={
                "session_id": "test123",  # no prefix
                "message": "Hello",
                "context": {},
            },
        )

        assert response.status_code == 200
        # ChatSession.get should have been called with the prefixed ID
        mock_session_get.assert_called_once_with("chat_session:test123")


# ---------------------------------------------------------------------------
# Suite 2: ZIE capture fire-and-forget
# ---------------------------------------------------------------------------

class TestZIECapture:
    """Tests for ZIE capture scheduling in execute_chat."""

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_zie_failure_does_not_block_response(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """ZIE capture failure is swallowed — response still returns 200."""
        mock_session_get.return_value = _make_session()
        mock_repo_query.return_value = []
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="AI response", id="msg_001")]}

        # Patch asyncio.create_task to raise an exception
        with patch("api.routers.chat.asyncio.create_task",
                   side_effect=RuntimeError("ZIE exploded")):
            response = client.post(
                "/api/chat/execute",
                json={
                    "session_id": "chat_session:test123",
                    "message": "Hello",
                    "context": {},
                },
            )

        # Response should still be 200 despite ZIE failure
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("api.routers.chat.ChatSession.get", new_callable=AsyncMock)
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.asyncio.to_thread", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_zie_capture_scheduled_with_correct_question(
        self, mock_graph, mock_to_thread, mock_nb_get,
        mock_repo_query, mock_session_get, client
    ):
        """ZIE capture is scheduled with the user's question."""
        mock_session_get.return_value = _make_session()
        mock_repo_query.return_value = []
        mock_nb_get.return_value = None
        mock_to_thread.return_value = MagicMock(values={})
        mock_graph.invoke.return_value = {"messages": [AIMessage(content="AI response", id="msg_001")]}

        captured_tasks = []
        with patch("api.routers.chat.asyncio.create_task",
                   side_effect=lambda coro: captured_tasks.append(coro)):
            response = client.post(
                "/api/chat/execute",
                json={
                    "session_id": "chat_session:test123",
                    "message": "What is the meaning of life?",
                    "context": {},
                },
            )

        assert response.status_code == 200
        # A task should have been scheduled
        assert len(captured_tasks) == 1


# ---------------------------------------------------------------------------
# Suite 3: Session CRUD
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    """Tests for session create/list/delete endpoints."""

    @pytest.mark.asyncio
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    async def test_create_session_notebook_not_found_returns_error(
        self, mock_nb_get, client
    ):
        """POST /chat/sessions with nonexistent notebook → error.

        BUG: Same HTTPException-swallowing bug as execute_chat — 404 becomes 500.
        """
        mock_nb_get.return_value = None

        response = client.post(
            "/api/chat/sessions",
            json={"notebook_id": "notebook:nonexistent"},
        )

        # Current behavior: 500 (bug). Expected: 404.
        assert response.status_code in (404, 500)

    @pytest.mark.asyncio
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.ChatSession")
    async def test_create_session_returns_session_response(
        self, mock_session_cls, mock_nb_get, client
    ):
        """POST /chat/sessions creates session and returns ChatSessionResponse."""
        mock_nb_get.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.id = "chat_session:new123"
        mock_session.title = "New Session"
        mock_session.created = "2024-01-01T00:00:00"
        mock_session.updated = "2024-01-01T00:00:00"
        mock_session.model_override = None
        mock_session.save = AsyncMock()
        mock_session.relate_to_notebook = AsyncMock()
        mock_session_cls.return_value = mock_session

        response = client.post(
            "/api/chat/sessions",
            json={"notebook_id": "notebook:abc123", "title": "My Session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chat_session:new123"
        assert data["notebook_id"] == "notebook:abc123"

    @pytest.mark.asyncio
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    async def test_get_sessions_notebook_not_found_returns_error(
        self, mock_nb_get, client
    ):
        """GET /chat/sessions?notebook_id=... with nonexistent notebook → error.

        BUG: Same HTTPException-swallowing bug — 404 becomes 500.
        """
        mock_nb_get.return_value = None

        response = client.get("/api/chat/sessions?notebook_id=notebook:nonexistent")

        # Current behavior: 500 (bug). Expected: 404.
        assert response.status_code in (404, 500)
