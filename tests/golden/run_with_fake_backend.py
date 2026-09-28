#!/usr/bin/env python3
"""Run a runner's main() with a deterministic fake model, recording every prompt it sends.

Used by make_goldens.py and test_runner_golden.py to prove that refactors keep the
prompts, tool replies and saved records byte-identical. No model is called.

Usage: python run_with_fake_backend.py {finance|retail} OUT_DIR -- <runner arguments>
"""

import json
import re
import sys
from pathlib import Path

ENGINEERING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINEERING))

CALLS = []


class FakeBackend:
    """Two steps per turn: call the market tool, then recommend the first three listed items."""

    def __init__(self, *args, **kwargs):
        self.model_id = kwargs.get("model_id", "fake")

    def initialize(self):
        pass

    def generate(self, messages, enable_thinking=False):
        CALLS.append([{k: m[k] for k in ("role", "content") if k in m} for m in messages])
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if len(messages) <= 2 and "MarketDataTool" in text or len(messages) <= 2 and "ProductDataTool" in text:
            tool = "ProductDataTool" if "ProductDataTool" in text else "MarketDataTool"
            return json.dumps({"thought": "look up data", "action": {"name": tool, "args": {"limit": 10}}})
        if len(messages) <= 2:  # a request outside the agent loop, e.g. eliciting estimates
            names = re.findall(r"\b[A-Z]{2,5}\b", text)
            return json.dumps({n: 3 for n in dict.fromkeys(names)})
        last = str(messages[-1].get("content", ""))
        picks = re.findall(r'"(?:symbol|name)": "([^"]+)"', last)[:3] or ["PG"]
        return json.dumps({"thought": "recommend", "final": {
            "risk_tolerance": "moderate", "ranked_products": picks, "rationale": "fixed",
            "memory_update": {"risk_tolerance": 1, "goal_indices": [0], "constraint_indices": [0]}}})

    def broadcast_object(self, obj, src=0):
        return obj

    def barrier(self):
        pass

    @property
    def is_main_process(self):
        return True

    def close(self):
        pass


def patch_backends():
    for module in list(sys.modules.values()):
        if module is not None and hasattr(module, "BedrockConverseBackend"):
            setattr(module, "BedrockConverseBackend", FakeBackend)


def main():
    domain, out = sys.argv[1], Path(sys.argv[2])
    rest = sys.argv[sys.argv.index("--") + 1:]
    import agent_system.api_backends  # noqa: F401
    runner = __import__("run_convfinre" if domain == "finance" else "run_taubench_pilot")
    patch_backends()
    sys.argv = [runner.__file__] + rest + ["--backend-type", "bedrock-converse", "--model-id", "fake",
                                           "--workers", "1", "--out-dir", str(out)]
    runner.main()
    (out / "calls.json").write_text(json.dumps(CALLS, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
