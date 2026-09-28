"""API-based LLM backends for OpenAI and AWS Bedrock Claude.

These implement the same interface as QwenHFBackend so they can be used
as drop-in replacements in ReActAgent and run_convfinre.py.

Required interface (from react_agent.py analysis):
  - generate(messages: List[Dict], enable_thinking: bool) -> str
  - broadcast_object(obj, src=0) -> obj
  - barrier() -> None
  - is_main_process -> bool
  - initialize() -> None
  - close() -> None
"""

import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Retry config for API rate limits / transient errors
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0  # seconds


def _retry_with_backoff(fn, max_retries=_MAX_RETRIES, initial_backoff=_INITIAL_BACKOFF):
    """Call *fn*() with exponential backoff on transient failures."""
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            exc_name = type(exc).__name__
            # Retry on rate-limit and transient server errors
            retriable = any(k in exc_name.lower() for k in ("rate", "timeout", "connection", "server"))
            if not retriable:
                retriable = "empty response" in str(exc).lower()
            if not retriable and hasattr(exc, "status_code"):
                retriable = getattr(exc, "status_code", 0) in (429, 500, 502, 503, 529)
            if not retriable or attempt == max_retries - 1:
                raise
            logger.warning(
                "Retriable error (attempt %d/%d, backoff %.1fs): %s",
                attempt + 1, max_retries, backoff, exc,
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


# ======================================================================
# OpenAI
# ======================================================================

class OpenAIBackend:
    """OpenAI API backend (GPT-4.1, GPT-4o, GPT-5.2, o3, etc.)."""

    def __init__(
        self,
        model_id: str = "gpt-5.2-chat-latest",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    def initialize(self) -> None:
        import openai

        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass --api-key."
            )
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)
        # Validate with a lightweight call
        logger.info("OpenAI backend initialized: model=%s", self.model_id)

    # Reasoning models (o-series, GPT-5.x) only accept temperature=1
    _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def _is_reasoning_model(self) -> bool:
        m = self.model_id.lower()
        return any(m.startswith(p) for p in self._REASONING_PREFIXES)

    def _needs_system_merge(self) -> bool:
        """Some models (e.g. Gemma via OpenRouter) don't support system role."""
        m = self.model_id.lower()
        return "gemma" in m

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        if self._client is None:
            self.initialize()
        assert self._client is not None

        def _call():
            msgs = list(messages)
            # Merge system into first user message for models that don't support it
            if self._needs_system_merge() and msgs and msgs[0].get("role") == "system":
                sys_content = msgs[0]["content"]
                msgs = msgs[1:]
                if msgs and msgs[0].get("role") == "user":
                    msgs[0] = {**msgs[0], "content": sys_content + "\n\n" + msgs[0]["content"]}
                else:
                    msgs.insert(0, {"role": "user", "content": sys_content})

            kwargs: Dict[str, Any] = {
                "model": self.model_id,
                "messages": msgs,
                "max_completion_tokens": self.max_tokens,
            }
            # Only add json_object format for models that support it
            if not self._needs_system_merge():
                kwargs["response_format"] = {"type": "json_object"}
            if not self._is_reasoning_model():
                kwargs["temperature"] = self._temperature if self._temperature is not None else 0
            resp = self._client.chat.completions.create(**kwargs)
            result = resp.choices[0].message.content or ""
            if not result:
                raise RuntimeError("Empty response from API")
            return result

        return _retry_with_backoff(_call)

    # -- Distributed stubs (API = single process) --------------------------

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj

    def barrier(self) -> None:
        pass

    @property
    def is_main_process(self) -> bool:
        return True

    def close(self) -> None:
        self._client = None


# ======================================================================
# Claude Code CLI
# ======================================================================


class ClaudeCodeBackend:
    """Claude Code CLI backend (uses `claude -p` subprocess).

    Runs Claude models via the Claude Code CLI with OAuth subscription.
    Supports Sonnet 4.6, Opus 4.6, Haiku 4.5 etc.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._temperature = temperature

    def initialize(self) -> None:
        import subprocess as _sp
        # Verify CLI is available
        try:
            result = _sp.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
            logger.info("Claude Code CLI backend initialized: model=%s, cli_version=%s",
                        self.model_id, result.stdout.strip())
        except FileNotFoundError:
            raise RuntimeError("Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        import subprocess as _sp
        import json as _json

        # Build the prompt: merge system + conversation into a single prompt string
        system_parts = []
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(f"[{role}]: {content}")

        prompt = "\n\n".join(user_parts)

        def _call():
            import tempfile as _tf

            cmd = [
                "claude", "-p",
                "--model", self.model_id,
                "--output-format", "json",
                "--max-turns", "1",
                "--tools", "",          # no built-in tools: the agent loop supplies its own
                "--strict-mcp-config",  # and no MCP servers from the user's settings
            ]

            # Use temp file for long system prompts (CLI arg has length limits)
            _sys_file = None
            if system_parts:
                _sys_file = _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
                _sys_file.write("\n\n".join(system_parts))
                _sys_file.close()
                cmd.extend(["--system-prompt-file", _sys_file.name])

            # Inherit the session's auth, but silence the OpenTelemetry console exporter,
            # which otherwise prints JSON blobs on stdout ahead of the result object.
            import os as _os
            env = {k: v for k, v in _os.environ.items() if not k.startswith("OTEL_")}
            env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "0"
            try:
                result = _sp.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=env,
                )
            finally:
                if _sys_file:
                    import os
                    os.unlink(_sys_file.name)

            def _last_json(text):
                """The CLI may print other JSON (telemetry) before the result object."""
                i = text.rfind('{"type":"result"')
                if i == -1:
                    i = text.rfind('{"duration_api_ms"')
                return _json.loads(text[i:] if i != -1 else text)

            if result.returncode != 0:
                stderr = result.stderr.strip()
                detail = stderr
                try:
                    detail = detail or str(_last_json(result.stdout).get("result", ""))
                except Exception:
                    detail = detail or result.stdout[-300:]
                if not detail:
                    detail = f"no stderr; stdout={result.stdout[:300]!r}"
                raise RuntimeError(f"Claude CLI error (rc={result.returncode}): {detail}")

            try:
                output = _last_json(result.stdout)
                if output.get("is_error"):
                    raise RuntimeError(f"Claude CLI returned error: {output.get('result', 'unknown')}")
                text = output.get("result", "")
            except _json.JSONDecodeError:
                text = result.stdout.strip()

            if not text:
                raise RuntimeError("Empty response from Claude CLI")

            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first line (```json) and last line (```)
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                elif lines[0].startswith("```"):
                    lines = lines[1:]
                text = "\n".join(lines)

            return text

        return _retry_with_backoff(_call)

    # -- Distributed stubs (CLI = single process) --------------------------

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj

    def barrier(self) -> None:
        pass

    @property
    def is_main_process(self) -> bool:
        return True

    def close(self) -> None:
        pass


# ======================================================================
# AWS Bedrock Claude
# ======================================================================

class BedrockClaudeBackend:
    """AWS Bedrock Claude backend (Sonnet 4.6, Opus 4.6, Haiku 4.5)."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        aws_region: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.aws_region = aws_region
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def initialize(self) -> None:
        from anthropic import AnthropicBedrock

        self._client = AnthropicBedrock(aws_region=self.aws_region)
        logger.info(
            "Bedrock Claude backend initialized: model=%s region=%s",
            self.model_id,
            self.aws_region,
        )

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        if self._client is None:
            self.initialize()
        assert self._client is not None

        # Claude Messages API: system prompt is a separate parameter,
        # not in the messages list.
        system_text = ""
        api_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                # Concatenate multiple system messages (shouldn't happen, but safe)
                if system_text:
                    system_text += "\n\n"
                system_text += msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Claude requires messages to start with a user message.
        # If the first message is assistant (shouldn't happen with ReAct),
        # we prepend a minimal user turn.
        if api_messages and api_messages[0]["role"] != "user":
            api_messages.insert(0, {"role": "user", "content": "Begin."})

        def _call():
            kwargs: Dict[str, Any] = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "messages": api_messages,
            }
            if system_text:
                kwargs["system"] = system_text
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature

            resp = self._client.messages.create(**kwargs)
            # Extract text from content blocks
            parts = []
            for block in resp.content:
                if block.type == "text":
                    parts.append(block.text)
            result = "\n".join(parts)
            if not result:
                raise RuntimeError("Empty response from API")
            return result

        return _retry_with_backoff(_call)

    # -- Distributed stubs -------------------------------------------------

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj

    def barrier(self) -> None:
        pass

    @property
    def is_main_process(self) -> bool:
        return True

    def close(self) -> None:
        self._client = None


# ======================================================================
# AWS Bedrock Converse (generic — Mistral, Llama, etc.)
# ======================================================================

class BedrockConverseBackend:
    """AWS Bedrock Converse API backend for non-Claude models (Mistral, Llama, etc.)."""

    def __init__(
        self,
        model_id: str = "mistral.ministral-3-14b-instruct-v1:0",
        aws_region: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.aws_region = aws_region
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def initialize(self) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=self.aws_region)
        logger.info(
            "Bedrock Converse backend initialized: model=%s region=%s",
            self.model_id,
            self.aws_region,
        )

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        if self._client is None:
            self.initialize()
        assert self._client is not None

        # Converse API: system prompt goes in 'system' param, not messages.
        system_parts: list = []
        api_messages: list = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                api_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        # Converse requires alternating user/assistant. Merge consecutive same-role.
        if api_messages and api_messages[0]["role"] != "user":
            api_messages.insert(0, {"role": "user", "content": [{"text": "Begin."}]})

        def _call():
            kwargs: Dict[str, Any] = {
                "modelId": self.model_id,
                "messages": api_messages,
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                },
            }
            if system_parts:
                kwargs["system"] = system_parts
            if self._temperature is not None:
                kwargs["inferenceConfig"]["temperature"] = self._temperature

            resp = self._client.converse(**kwargs)
            # Extract text from output content blocks
            parts = []
            for block in resp.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    parts.append(block["text"])
            result = "\n".join(parts)
            if not result:
                raise RuntimeError("Empty response from Bedrock Converse API")
            return result

        return _retry_with_backoff(_call)

    # -- Distributed stubs -------------------------------------------------

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj

    def barrier(self) -> None:
        pass

    @property
    def is_main_process(self) -> bool:
        return True

    def close(self) -> None:
        self._client = None
