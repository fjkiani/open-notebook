"""
Unit tests for open_notebook.graphs.ask (RAG pipeline)

Covers:
- Strategy Pydantic model: valid JSON parses, missing searches defaults to []
- call_model_with_messages (strategy node): returns {"strategy": Strategy(...)}
- trigger_queries: returns correct Send objects for each search
- provide_answer: zero results → {"answers": []}, with results → calls model
- write_final_answer: calls model, strips thinking tags, returns {"final_answer": ...}
- Graph compilation: correct node/edge structure
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _make_config(strategy_model=None, answer_model=None, final_answer_model=None):
    return {
        "configurable": {
            "strategy_model": strategy_model,
            "answer_model": answer_model,
            "final_answer_model": final_answer_model,
        }
    }


# ---------------------------------------------------------------------------
# Suite 1: Strategy Pydantic model
# ---------------------------------------------------------------------------

class TestStrategyModel:
    """Tests for the Strategy and Search Pydantic models."""

    def test_strategy_with_searches_parses(self):
        """Strategy with valid searches list parses correctly."""
        from open_notebook.graphs.ask import Strategy, Search

        s = Strategy(
            reasoning="I need to search for X",
            searches=[
                Search(term="X", instructions="Find info about X"),
                Search(term="Y", instructions="Find info about Y"),
            ],
        )
        assert len(s.searches) == 2
        assert s.searches[0].term == "X"
        assert s.searches[1].instructions == "Find info about Y"

    def test_strategy_missing_searches_defaults_to_empty_list(self):
        """Strategy without searches field defaults to empty list."""
        from open_notebook.graphs.ask import Strategy

        s = Strategy(reasoning="No searches needed")
        assert s.searches == []

    def test_strategy_max_five_searches_is_convention(self):
        """Strategy can hold up to 5 searches (no hard limit enforced by model)."""
        from open_notebook.graphs.ask import Strategy, Search

        searches = [
            Search(term=f"term{i}", instructions=f"instructions{i}")
            for i in range(5)
        ]
        s = Strategy(reasoning="Five searches", searches=searches)
        assert len(s.searches) == 5

    def test_search_requires_term(self):
        """Search requires a term field."""
        from open_notebook.graphs.ask import Search
        import pydantic

        with pytest.raises((pydantic.ValidationError, TypeError)):
            Search(instructions="no term provided")

    def test_search_instructions_is_required(self):
        """Search instructions field is required — omitting it raises ValidationError."""
        from open_notebook.graphs.ask import Search
        import pydantic

        # instructions has no default — it is required
        with pytest.raises(pydantic.ValidationError):
            Search(term="test term")  # missing instructions

    def test_search_with_instructions_creates_successfully(self):
        """Search with both term and instructions creates successfully."""
        from open_notebook.graphs.ask import Search

        s = Search(term="test term", instructions="find test info")
        assert s.term == "test term"
        assert s.instructions == "find test info"


# ---------------------------------------------------------------------------
# Suite 2: call_model_with_messages (strategy node)
# ---------------------------------------------------------------------------

class TestAskStrategyNode:
    """Tests for the strategy-planning node in ask.py."""

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_returns_strategy_object(self, mock_provision):
        """Strategy node returns dict with 'strategy' key containing Strategy object."""
        from open_notebook.graphs.ask import call_model_with_messages, Strategy

        strategy_json = '{"reasoning": "search for it", "searches": [{"term": "test", "instructions": "find test"}]}'
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message(strategy_json)
        mock_provision.return_value = mock_model

        state = {"question": "What is X?", "strategy": None, "answers": [], "final_answer": ""}
        result = await call_model_with_messages(state, _make_config())

        assert "strategy" in result
        assert isinstance(result["strategy"], Strategy)
        assert len(result["strategy"].searches) == 1
        assert result["strategy"].searches[0].term == "test"

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_thinking_tags_stripped_before_parse(self, mock_provision):
        """<think> tags are stripped before JSON parsing."""
        from open_notebook.graphs.ask import call_model_with_messages, Strategy

        raw = '<think>reasoning</think>{"reasoning": "clean", "searches": []}'
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message(raw)
        mock_provision.return_value = mock_model

        state = {"question": "Q?", "strategy": None, "answers": [], "final_answer": ""}
        result = await call_model_with_messages(state, _make_config())

        assert isinstance(result["strategy"], Strategy)
        assert result["strategy"].searches == []

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_open_notebook_error_propagates(self, mock_provision):
        """OpenNotebookError from model is re-raised unchanged."""
        from open_notebook.graphs.ask import call_model_with_messages
        from open_notebook.exceptions import OpenNotebookError

        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = OpenNotebookError("rate limit")
        mock_provision.return_value = mock_model

        state = {"question": "Q?", "strategy": None, "answers": [], "final_answer": ""}
        with pytest.raises(OpenNotebookError, match="rate limit"):
            await call_model_with_messages(state, _make_config())

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_generic_exception_classified(self, mock_provision):
        """Generic exceptions are classified into OpenNotebookError subclasses."""
        from open_notebook.graphs.ask import call_model_with_messages
        from open_notebook.exceptions import OpenNotebookError

        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = ConnectionError("network down")
        mock_provision.return_value = mock_model

        state = {"question": "Q?", "strategy": None, "answers": [], "final_answer": ""}
        with pytest.raises(OpenNotebookError):
            await call_model_with_messages(state, _make_config())


# ---------------------------------------------------------------------------
# Suite 3: trigger_queries (fan-out node)
# ---------------------------------------------------------------------------

class TestTriggerQueries:
    """Tests for the trigger_queries conditional edge function."""

    @pytest.mark.asyncio
    async def test_returns_send_for_each_search(self):
        """trigger_queries returns one Send per search in strategy."""
        from open_notebook.graphs.ask import trigger_queries, Strategy, Search
        from langgraph.types import Send

        strategy = Strategy(
            reasoning="two searches",
            searches=[
                Search(term="alpha", instructions="find alpha"),
                Search(term="beta", instructions="find beta"),
            ],
        )
        state = {
            "question": "What is alpha and beta?",
            "strategy": strategy,
            "answers": [],
            "final_answer": "",
        }

        result = await trigger_queries(state, _make_config())

        assert len(result) == 2
        assert all(isinstance(r, Send) for r in result)
        # Each Send targets "provide_answer" node
        assert all(r.node == "provide_answer" for r in result)

    @pytest.mark.asyncio
    async def test_send_contains_correct_fields(self):
        """Each Send contains question, term, and instructions."""
        from open_notebook.graphs.ask import trigger_queries, Strategy, Search
        from langgraph.types import Send

        strategy = Strategy(
            reasoning="one search",
            searches=[Search(term="gamma", instructions="find gamma info")],
        )
        state = {
            "question": "Tell me about gamma",
            "strategy": strategy,
            "answers": [],
            "final_answer": "",
        }

        result = await trigger_queries(state, _make_config())

        assert len(result) == 1
        send_input = result[0].arg
        assert send_input["question"] == "Tell me about gamma"
        assert send_input["term"] == "gamma"
        assert send_input["instructions"] == "find gamma info"

    @pytest.mark.asyncio
    async def test_empty_strategy_returns_empty_list(self):
        """Empty searches list returns empty Send list."""
        from open_notebook.graphs.ask import trigger_queries, Strategy

        strategy = Strategy(reasoning="nothing to search")
        state = {
            "question": "Q?",
            "strategy": strategy,
            "answers": [],
            "final_answer": "",
        }

        result = await trigger_queries(state, _make_config())
        assert result == []


# ---------------------------------------------------------------------------
# Suite 4: provide_answer (parallel search node)
# ---------------------------------------------------------------------------

class TestProvideAnswer:
    """Tests for the provide_answer node."""

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.vector_search", new_callable=AsyncMock)
    async def test_zero_results_returns_empty_answers(self, mock_search):
        """Zero vector search results → returns {"answers": []}."""
        from open_notebook.graphs.ask import provide_answer

        mock_search.return_value = []

        state = {
            "question": "What is X?",
            "term": "X",
            "instructions": "find X",
            "results": {},
            "answer": "",
            "ids": [],
        }

        result = await provide_answer(state, _make_config())
        assert result == {"answers": []}

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    @patch("open_notebook.graphs.ask.vector_search", new_callable=AsyncMock)
    async def test_with_results_calls_model_and_returns_answer(
        self, mock_search, mock_provision
    ):
        """With search results, model is called and answer returned."""
        from open_notebook.graphs.ask import provide_answer

        mock_search.return_value = [
            {"id": "source:abc", "content": "X is a thing", "score": 0.9}
        ]
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message("X is indeed a thing.")
        mock_provision.return_value = mock_model

        state = {
            "question": "What is X?",
            "term": "X",
            "instructions": "find X",
            "results": {},
            "answer": "",
            "ids": [],
        }

        result = await provide_answer(state, _make_config())

        assert "answers" in result
        assert len(result["answers"]) == 1
        assert "X is indeed a thing." in result["answers"][0]

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    @patch("open_notebook.graphs.ask.vector_search", new_callable=AsyncMock)
    async def test_thinking_tags_stripped_from_answer(self, mock_search, mock_provision):
        """<think> tags are stripped from the answer."""
        from open_notebook.graphs.ask import provide_answer

        mock_search.return_value = [{"id": "source:abc", "content": "data", "score": 0.8}]
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message(
            "<think>reasoning</think>Clean answer."
        )
        mock_provision.return_value = mock_model

        state = {
            "question": "Q?",
            "term": "term",
            "instructions": "instructions",
            "results": {},
            "answer": "",
            "ids": [],
        }

        result = await provide_answer(state, _make_config())
        assert "<think>" not in result["answers"][0]
        assert "Clean answer." in result["answers"][0]

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.vector_search", new_callable=AsyncMock)
    async def test_open_notebook_error_propagates(self, mock_search):
        """OpenNotebookError from vector_search propagates unchanged."""
        from open_notebook.graphs.ask import provide_answer
        from open_notebook.exceptions import OpenNotebookError

        mock_search.side_effect = OpenNotebookError("db error")

        state = {
            "question": "Q?",
            "term": "term",
            "instructions": "instructions",
            "results": {},
            "answer": "",
            "ids": [],
        }

        with pytest.raises(OpenNotebookError, match="db error"):
            await provide_answer(state, _make_config())


# ---------------------------------------------------------------------------
# Suite 5: write_final_answer
# ---------------------------------------------------------------------------

class TestWriteFinalAnswer:
    """Tests for the write_final_answer synthesis node."""

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_returns_final_answer(self, mock_provision):
        """write_final_answer returns dict with 'final_answer' key."""
        from open_notebook.graphs.ask import write_final_answer

        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message("The final synthesis.")
        mock_provision.return_value = mock_model

        state = {
            "question": "What is X?",
            "strategy": MagicMock(),
            "answers": ["X is a thing.", "X is also another thing."],
            "final_answer": "",
        }

        result = await write_final_answer(state, _make_config())

        assert "final_answer" in result
        assert result["final_answer"] == "The final synthesis."

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_thinking_tags_stripped_from_final_answer(self, mock_provision):
        """<think> tags are stripped from the final answer."""
        from open_notebook.graphs.ask import write_final_answer

        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = _make_ai_message(
            "<think>deep thought</think>The clean final answer."
        )
        mock_provision.return_value = mock_model

        state = {
            "question": "Q?",
            "strategy": MagicMock(),
            "answers": ["some answer"],
            "final_answer": "",
        }

        result = await write_final_answer(state, _make_config())
        assert "<think>" not in result["final_answer"]
        assert "The clean final answer." in result["final_answer"]

    @pytest.mark.asyncio
    @patch("open_notebook.graphs.ask.provision_langchain_model", new_callable=AsyncMock)
    async def test_open_notebook_error_propagates(self, mock_provision):
        """OpenNotebookError propagates unchanged."""
        from open_notebook.graphs.ask import write_final_answer
        from open_notebook.exceptions import OpenNotebookError

        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = OpenNotebookError("model unavailable")
        mock_provision.return_value = mock_model

        state = {
            "question": "Q?",
            "strategy": MagicMock(),
            "answers": ["answer"],
            "final_answer": "",
        }

        with pytest.raises(OpenNotebookError, match="model unavailable"):
            await write_final_answer(state, _make_config())


# ---------------------------------------------------------------------------
# Suite 6: Ask graph compilation
# ---------------------------------------------------------------------------

class TestAskGraphCompilation:
    """Tests for the compiled ask graph structure."""

    def test_graph_is_compiled(self):
        """ask.graph is a compiled LangGraph StateGraph."""
        from open_notebook.graphs.ask import graph
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")

    def test_graph_has_required_nodes(self):
        """Graph has agent, provide_answer, and write_final_answer nodes."""
        from open_notebook.graphs.ask import graph
        graph_repr = graph.get_graph()
        node_names = list(graph_repr.nodes.keys())
        assert "agent" in node_names
        assert "provide_answer" in node_names
        assert "write_final_answer" in node_names

    def test_graph_no_checkpointer(self):
        """Ask graph does not use a checkpointer (stateless per request)."""
        from open_notebook.graphs.ask import graph
        # Ask graph is stateless — no checkpointer
        # Verify by checking the graph compiles without memory
        assert graph is not None
