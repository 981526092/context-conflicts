"""One multi-turn session: the loop both runners share.

Each turn builds fresh tools (the data changes per turn), a fresh ReActAgent and the
user's message; the memory object persists across turns. The domain supplies the tool
builder, the user message and the fields attached to each saved turn.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from .react_agent import ReActAgent
from .tools import Tool

logger = logging.getLogger(__name__)

FROZEN_FIELDS = ("risk_tolerance", "goals", "constraints")


def run_session(
    llm,
    *,
    user_id: str,
    label: str,
    memory,
    steps: Sequence[Any],
    build_tools: Callable[[int, Any], List[Tool]],
    user_message: Callable[[Any], str],
    annotate: Callable[[Dict[str, Any], Any], None],
    max_steps: int,
    enable_thinking: bool,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    frozen_memory: Optional[List[Dict[str, Any]]] = None,
    log_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``steps`` as consecutive turns and return the saved session record.

    With ``frozen_memory``, turn i starts from ``frozen_memory[i-1]`` and the agent may not
    write memory, so memory is set from outside instead of by the agent.
    """
    initial_memory = memory.snapshot()
    frozen = frozen_memory is not None
    results: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for i, step in enumerate(steps, start=1):
        expected = None
        if frozen and i - 1 < len(frozen_memory):
            expected = frozen_memory[i - 1]
            memory.load_snapshot(expected)

        agent = ReActAgent(llm=llm, tools=build_tools(i, step), max_steps=max_steps,
                           freeze_memory=frozen, **(agent_kwargs or {}))
        message = user_message(step)
        if llm.is_main_process:
            logger.info("[%s/%s] Step %d/%d: %s", label, user_id, i,
                        log_total or len(steps), message[:80])

        result = agent.run_turn(turn_index=i, user_input=message, memory=memory,
                                enable_thinking=enable_thinking)

        if expected is not None:
            after = memory.snapshot()
            changed = [f for f in FROZEN_FIELDS if after[f] != expected[f]]
            assert not changed, f"frozen memory changed at turn {i}, user {user_id}: {changed}"

        annotate(result, step)
        results.append(result)
        traces.extend(agent.export_trace())

    return {
        "label": label,
        "user_id": user_id,
        "initial_memory": initial_memory,
        "results": results,
        "trace": traces,
    }
