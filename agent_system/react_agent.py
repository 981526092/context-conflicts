import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .memory import MemoryState
from .tools import Tool
from .utils import extract_first_json_object

logger = logging.getLogger(__name__)


@dataclass
class ReActTraceStep:
    turn: int
    step: int
    llm_raw: str
    thought: str
    action: Optional[Dict[str, Any]]
    observation: Optional[Dict[str, Any]]
    final: Optional[Dict[str, Any]]


_RISK_BAND_MAP = {"low": 2, "moderate": 3, "high": 5}


class ReActAgent:
    def __init__(
        self,
        llm,
        tools: List[Tool],
        max_steps: int = 6,
        system_prompt_override: Optional[str] = None,
        system_prompt_suffix: Optional[str] = None,
        guardrail_mode: bool = False,
        stock_risk_db: Optional[Dict[str, int]] = None,
        freeze_memory: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self._system_prompt_override = system_prompt_override
        self._system_prompt_suffix = system_prompt_suffix or ""
        self.trace: List[ReActTraceStep] = []
        self._guardrail_mode = guardrail_mode
        self._stock_risk_db = stock_risk_db or {}
        self._freeze_memory = freeze_memory

    def _tool_spec_text(self) -> str:
        lines = []
        for tool in self.tools.values():
            lines.append(
                f"- {tool.name}: {tool.description}. args={tool.arg_schema if tool.arg_schema else '{}'}"
            )
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        if self._system_prompt_override:
            return self._system_prompt_override + f"\n\nAvailable tools:\n{self._tool_spec_text()}"
        return (
            "You are a financial recommendation agent following ReAct orchestration.\n"
            "\n"
            "WORKFLOW (follow this order every turn):\n"
            "1. Call MarketDataTool to retrieve current stock candidates and metrics.\n"
            "2. Optionally call NewsRetrieverTool for recent market sentiment.\n"
            "3. Analyze the data together with the user's message and memory profile.\n"
            "4. Produce your final recommendation based on tool results.\n"
            "\n"
            "IMPORTANT: You MUST call at least one tool before producing a final answer.\n"
            "Do NOT finalize without consulting current market data.\n"
            "\n"
            "OUTPUT FORMAT: return one JSON object per response.\n"
            "To call a tool:\n"
            '{"thought":"...", "action":{"name":"ToolName","args":{...}}}\n'
            "To finalize (only after gathering data):\n"
            '{"thought":"...", "final":{"risk_tolerance":"low|moderate|high",'
            '"ranked_products":["TICKER1","TICKER2",...],"rationale":"...",'
            '"memory_update":{"risk_tolerance":0,"goal_indices":[0,3],"constraint_indices":[0]}}}\n'
            f"\n{MemoryState.options_text()}\n"
            "Use integer indices only for memory_update fields.\n"
            "\n"
            "CONSTRAINTS:\n"
            '- "ranked_products" must contain ONLY ticker symbols from MarketDataTool results.\n'
            "- Do not add prose outside JSON.\n"
            "\n"
            f"Available tools:\n{self._tool_spec_text()}"
            f"{self._system_prompt_suffix}"
        )

    def _build_initial_prompt(self, user_input: str, memory: MemoryState) -> str:
        """Build the first user message for a turn (step 1 only)."""
        return (
            "User request:\n"
            f"{user_input}\n\n"
            "Current memory profile:\n"
            f"{memory.snapshot()}\n\n"
            "Start by calling MarketDataTool to get current stock candidates."
        )

    def _build_observation_prompt(self, last_step: Dict[str, Any]) -> str:
        """Build a follow-up user message containing only the latest observation."""
        return (
            "Observation from previous action:\n"
            f"{json.dumps(last_step, ensure_ascii=False)}\n\n"
            "Based on the data above, call another tool or finalize your recommendation."
        )

    def _build_parse_error_prompt(self, raw: str) -> str:
        """Build feedback when LLM output fails JSON parsing."""
        snippet = raw[:200] + ("..." if len(raw) > 200 else "")
        return (
            "JSON parse failure. Your previous response could not be parsed:\n"
            f"{snippet}\n\n"
            "Please respond with a valid JSON object matching either the action or final schema."
        )

    def _orchestrate_step(
        self,
        raw: str,
        turn_index: int,
        step_index: int,
        memory: MemoryState,
    ) -> Dict[str, Any]:
        """Parse LLM output, execute tools, update memory (rank 0 only).

        Returns a decision dict broadcast to all ranks:
        - {"type": "continue", "thought", "action", "observation", "raw"}
        - {"type": "final", "thought", "final", "raw"}
        - {"type": "parse_error", "raw"}
        """
        parsed = extract_first_json_object(raw) or {}
        thought = str(parsed.get("thought", ""))
        action = parsed.get("action")
        final = parsed.get("final")

        # --- Tool call ---
        if isinstance(action, dict) and "name" in action:
            tool_name = str(action.get("name"))
            tool_args = action.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            if tool_name in self.tools:
                observation = self.tools[tool_name].run(tool_args, memory)
            else:
                observation = {"error": f"unknown tool: {tool_name}"}

            self.trace.append(ReActTraceStep(
                turn=turn_index,
                step=step_index,
                llm_raw=raw,
                thought=thought,
                action={"name": tool_name, "args": tool_args},
                observation=observation,
                final=None,
            ))
            return {
                "type": "continue",
                "thought": thought,
                "action": {"name": tool_name, "args": tool_args},
                "observation": observation,
                "raw": raw,
            }

        # --- Final answer ---
        if isinstance(final, dict):
            if not self._freeze_memory:
                proposed_update = final.get("memory_update", {})
                if not isinstance(proposed_update, dict):
                    proposed_update = {}
                memory.apply_update(proposed_update)

            ranked = final.get("ranked_products", [])
            if isinstance(ranked, list):
                # Apply suitability guardrail: filter products exceeding risk band
                if self._guardrail_mode and self._stock_risk_db:
                    band = _RISK_BAND_MAP.get(memory.risk_tolerance.lower(), 3)
                    filtered = [
                        p for p in ranked
                        if self._stock_risk_db.get(str(p), 5) <= band
                    ]
                    if filtered:
                        ranked = filtered
                        final["ranked_products"] = ranked
                        final["guardrail_applied"] = True
                    else:
                        # All products violate — keep original to avoid empty list
                        final["guardrail_applied"] = False
                if not self._freeze_memory:
                    memory.recent_decisions = [str(x) for x in ranked[:5]]

            self.trace.append(ReActTraceStep(
                turn=turn_index,
                step=step_index,
                llm_raw=raw,
                thought=thought,
                action=action if isinstance(action, dict) else None,
                observation=None,
                final=final,
            ))
            return {
                "type": "final",
                "thought": thought,
                "final": final,
                "memory_snapshot": memory.snapshot(),
                "raw": raw,
            }

        # --- Parse failure ---
        logger.warning("Step %d/%d: no valid action or final in LLM output", turn_index, step_index)
        self.trace.append(ReActTraceStep(
            turn=turn_index,
            step=step_index,
            llm_raw=raw,
            thought=thought,
            action=None,
            observation=None,
            final=None,
        ))
        return {"type": "parse_error", "raw": raw}

    def run_turn(
        self,
        turn_index: int,
        user_input: str,
        memory: MemoryState,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        # Sync memory snapshot at turn start so all ranks agree
        memory_snap = memory.snapshot()
        memory_snap = self.llm.broadcast_object(memory_snap, src=0)
        if not self.llm.is_main_process:
            memory.load_snapshot(memory_snap)

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        # Pre-set the first user message; subsequent ones are set at the end of each iteration
        next_user_msg = self._build_initial_prompt(user_input, memory)

        for step_index in range(1, self.max_steps + 1):
            messages.append({"role": "user", "content": next_user_msg})

            # ALL ranks call generate (required for tensor parallelism)
            raw = self.llm.generate(messages, enable_thinking=enable_thinking)

            # Rank 0 orchestrates; others wait for broadcast
            if self.llm.is_main_process:
                decision = self._orchestrate_step(raw, turn_index, step_index, memory)
            else:
                decision = None

            decision = self.llm.broadcast_object(decision, src=0)

            # --- Handle decision on all ranks ---
            if decision["type"] == "final":
                # Sync rank 0's post-update memory to all ranks
                if not self.llm.is_main_process:
                    memory.load_snapshot(decision["memory_snapshot"])

                return {
                    "turn": turn_index,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "user_input": user_input,
                    "final": decision["final"],
                    "memory_after": memory.snapshot(),
                }

            if decision["type"] == "continue":
                # All ranks update their message history identically
                messages.append({"role": "assistant", "content": decision["raw"]})
                next_user_msg = self._build_observation_prompt({
                    "step": step_index,
                    "thought": decision["thought"],
                    "action": decision["action"],
                    "observation": decision["observation"],
                })
                continue

            if decision["type"] == "parse_error":
                messages.append({"role": "assistant", "content": decision["raw"]})
                next_user_msg = self._build_parse_error_prompt(decision["raw"])
                continue

        # Max steps exhausted — record as failed turn, memory unchanged
        return {
            "turn": turn_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input,
            "final": None,
            "memory_after": memory.snapshot(),
        }

    def export_trace(self) -> List[Dict[str, Any]]:
        return [asdict(step) for step in self.trace]
