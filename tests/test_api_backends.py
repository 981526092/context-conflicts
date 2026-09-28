#!/usr/bin/env python3
"""Tests for API backends (OpenAI + Bedrock Claude).

Uses unittest.mock to avoid real API calls. Validates:
  - Interface compatibility with QwenHFBackend
  - Message format handling (system prompt extraction for Claude)
  - Retry logic on transient errors
  - Integration with ReActAgent
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.api_backends import (
    BedrockClaudeBackend,
    OpenAIBackend,
    _retry_with_backoff,
)
from agent_system.memory import MemoryState
from agent_system.react_agent import ReActAgent
from agent_system.tools import Tool


# ======================================================================
# Helper: minimal tool for agent integration test
# ======================================================================
class DummyTool(Tool):
    def __init__(self):
        super().__init__(
            name="DummyTool",
            description="Returns dummy data",
            arg_schema={},
        )

    def run(self, args, memory):
        return {"data": "dummy"}


# ======================================================================
# OpenAIBackend tests
# ======================================================================
class TestOpenAIBackend(unittest.TestCase):

    def test_interface_stubs(self):
        backend = OpenAIBackend(model_id="gpt-4o", api_key="test-key")
        self.assertTrue(backend.is_main_process)
        self.assertEqual(backend.broadcast_object({"x": 1}), {"x": 1})
        backend.barrier()  # no-op
        backend.close()

    def test_initialize_sets_client(self):
        mock_openai_mod = MagicMock()
        mock_client_cls = mock_openai_mod.OpenAI
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            backend = OpenAIBackend(model_id="gpt-4o", api_key="sk-test")
            backend.initialize()

        mock_client_cls.assert_called_once_with(api_key="sk-test")
        self.assertIsNotNone(backend._client)

    def test_initialize_raises_without_key(self):
        backend = OpenAIBackend(model_id="gpt-4o", api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if present
            backend._api_key = None
            with self.assertRaises(ValueError):
                backend.initialize()

    def test_generate_calls_api(self):
        backend = OpenAIBackend(model_id="gpt-4o", api_key="sk-test")

        # Mock the client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"thought":"test","final":{"ranked_products":["SPY"]}}'
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        messages = [
            {"role": "system", "content": "You are a financial agent."},
            {"role": "user", "content": "Recommend stocks."},
        ]
        result = backend.generate(messages)

        self.assertIn("thought", result)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "gpt-4o")
        self.assertEqual(call_kwargs.kwargs["messages"], messages)
        self.assertEqual(call_kwargs.kwargs["temperature"], 0)

    def test_generate_raises_on_none_content(self):
        backend = OpenAIBackend(model_id="gpt-4o", api_key="sk-test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        with self.assertRaises(RuntimeError):
            backend.generate([{"role": "user", "content": "hi"}])


# ======================================================================
# BedrockClaudeBackend tests
# ======================================================================
class TestBedrockClaudeBackend(unittest.TestCase):

    def test_interface_stubs(self):
        backend = BedrockClaudeBackend(model_id="us.anthropic.claude-sonnet-4-6")
        self.assertTrue(backend.is_main_process)
        self.assertEqual(backend.broadcast_object([1, 2, 3]), [1, 2, 3])
        backend.barrier()
        backend.close()

    def test_generate_extracts_system_prompt(self):
        """Verify system messages are separated from the messages list."""
        backend = BedrockClaudeBackend(model_id="us.anthropic.claude-sonnet-4-6")

        mock_client = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"thought":"ok","final":{"ranked_products":["VTI"]}}'
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        messages = [
            {"role": "system", "content": "You are a financial agent."},
            {"role": "user", "content": "Recommend ETFs."},
            {"role": "assistant", "content": '{"thought":"looking up"}'},
            {"role": "user", "content": "Continue."},
        ]
        result = backend.generate(messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # System should be extracted as separate param
        self.assertEqual(call_kwargs["system"], "You are a financial agent.")
        # Messages should NOT contain system role
        api_msgs = call_kwargs["messages"]
        for msg in api_msgs:
            self.assertNotEqual(msg["role"], "system")
        # First message should be user
        self.assertEqual(api_msgs[0]["role"], "user")
        self.assertIn("thought", result)

    def test_generate_handles_no_system(self):
        """Messages without system role should work (no system kwarg)."""
        backend = BedrockClaudeBackend(model_id="us.anthropic.claude-sonnet-4-6")

        mock_client = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "hello"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        messages = [{"role": "user", "content": "Hi"}]
        backend.generate(messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertNotIn("system", call_kwargs)


# ======================================================================
# Retry logic tests
# ======================================================================
class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_on_first_try(self):
        fn = MagicMock(return_value="ok")
        result = _retry_with_backoff(fn, max_retries=3, initial_backoff=0.01)
        self.assertEqual(result, "ok")
        fn.assert_called_once()

    def test_retries_on_rate_limit(self):
        exc = type("RateLimitError", (Exception,), {})()
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        result = _retry_with_backoff(fn, max_retries=3, initial_backoff=0.01)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.call_count, 3)

    def test_raises_non_retriable(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with self.assertRaises(ValueError):
            _retry_with_backoff(fn, max_retries=3, initial_backoff=0.01)
        fn.assert_called_once()

    def test_retries_on_status_code_429(self):
        exc = Exception("too many requests")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, "ok"])
        result = _retry_with_backoff(fn, max_retries=3, initial_backoff=0.01)
        self.assertEqual(result, "ok")

    def test_exhausts_retries(self):
        exc = type("RateLimitError", (Exception,), {})()
        fn = MagicMock(side_effect=exc)
        with self.assertRaises(Exception):
            _retry_with_backoff(fn, max_retries=2, initial_backoff=0.01)
        self.assertEqual(fn.call_count, 2)


# ======================================================================
# ReActAgent integration test (mock backend)
# ======================================================================
class TestReActAgentWithAPIBackend(unittest.TestCase):
    """Verify ReActAgent works with API backends (not just QwenHFBackend)."""

    def _make_mock_backend(self, responses):
        """Create a mock backend that returns canned responses."""
        backend = MagicMock()
        backend.is_main_process = True
        backend.generate = MagicMock(side_effect=responses)
        backend.broadcast_object = lambda obj, src=0: obj
        backend.barrier = MagicMock()
        return backend

    def test_single_tool_then_final(self):
        """Agent should call tool, get observation, then produce final answer."""
        tool_response = json.dumps({
            "thought": "Let me look up market data.",
            "action": {"name": "DummyTool", "args": {}},
        })
        final_response = json.dumps({
            "thought": "Based on data, recommending SPY.",
            "final": {
                "risk_tolerance": "moderate",
                "ranked_products": ["SPY", "VTI", "BND"],
                "rationale": "Diversified moderate-risk portfolio",
                "memory_update": {},
            },
        })

        backend = self._make_mock_backend([tool_response, final_response])
        tool = DummyTool()
        agent = ReActAgent(llm=backend, tools=[tool], max_steps=6)
        memory = MemoryState()

        result = agent.run_turn(turn_index=1, user_input="Recommend stocks", memory=memory)

        self.assertIsNotNone(result["final"])
        self.assertEqual(result["final"]["ranked_products"], ["SPY", "VTI", "BND"])
        self.assertEqual(backend.generate.call_count, 2)

    def test_direct_final_no_tools(self):
        """Agent should handle immediate final answer."""
        final_response = json.dumps({
            "thought": "User wants conservative stocks.",
            "final": {
                "risk_tolerance": "low",
                "ranked_products": ["BND", "VTI"],
                "rationale": "Conservative portfolio",
                "memory_update": {"risk_tolerance": 0},
            },
        })

        backend = self._make_mock_backend([final_response])
        agent = ReActAgent(llm=backend, tools=[], max_steps=6)
        memory = MemoryState()

        result = agent.run_turn(turn_index=1, user_input="Conservative recs", memory=memory)

        self.assertIsNotNone(result["final"])
        self.assertEqual(backend.generate.call_count, 1)

    def test_parse_error_recovery(self):
        """Agent should recover from a parse error on first attempt."""
        bad_response = "I'm not sure what format to use..."
        good_response = json.dumps({
            "thought": "Now using correct format.",
            "final": {
                "risk_tolerance": "moderate",
                "ranked_products": ["SPY"],
                "rationale": "Fixed",
                "memory_update": {},
            },
        })

        backend = self._make_mock_backend([bad_response, good_response])
        agent = ReActAgent(llm=backend, tools=[], max_steps=6)
        memory = MemoryState()

        result = agent.run_turn(turn_index=1, user_input="Test", memory=memory)

        self.assertIsNotNone(result["final"])
        self.assertEqual(backend.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
