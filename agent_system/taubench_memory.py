"""Retail domain memory state for τ-bench pilot experiments.

Mirrors the MemoryState API exactly so that metrics.py functions
(memory_drift_rate, goal_drift_rate) work without modification.

Key alignment: snapshot() outputs ``risk_tolerance`` (not ``budget_sensitivity``)
so that memory_drift_rate() comparisons work transparently.
"""

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .taubench_dataset import RetailStep


BUDGET_ORDER = ["tight", "moderate", "flexible"]

GOAL_OPTIONS = [
    "daily essentials",
    "home improvement",
    "fitness gear",
    "tech upgrade",
    "gift shopping",
    "fashion update",
    "hobby supplies",
]

CONSTRAINT_OPTIONS = [
    "strict monthly budget",
    "limited storage space",
    "eco-friendly preference",
    "brand loyalty",
]


@dataclass
class RetailMemoryState:
    budget_sensitivity: str = "moderate"
    goals: List[str] = field(default_factory=lambda: ["daily essentials"])
    constraints: List[str] = field(default_factory=lambda: ["strict monthly budget"])
    recent_decisions: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, object]:
        # Map budget_sensitivity → risk_tolerance for metric compatibility
        return {
            "risk_tolerance": self.budget_sensitivity,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "recent_decisions": list(self.recent_decisions),
        }

    def load_snapshot(self, snap: Dict[str, object]) -> None:
        if "risk_tolerance" in snap and str(snap["risk_tolerance"]) in BUDGET_ORDER:
            self.budget_sensitivity = str(snap["risk_tolerance"])
        if "goals" in snap and isinstance(snap["goals"], list):
            self.goals = [str(x) for x in snap["goals"]][:10]
        if "constraints" in snap and isinstance(snap["constraints"], list):
            self.constraints = [str(x) for x in snap["constraints"]][:10]
        if "recent_decisions" in snap and isinstance(snap["recent_decisions"], list):
            self.recent_decisions = [str(x) for x in snap["recent_decisions"]][:10]

    def apply_update(self, updates: Dict[str, object]) -> None:
        # budget_sensitivity (mapped as risk_tolerance): index only
        if "risk_tolerance" in updates:
            val = updates["risk_tolerance"]
            if isinstance(val, int) and 0 <= val < len(BUDGET_ORDER):
                self.budget_sensitivity = BUDGET_ORDER[val]

        # goals: index list only
        if "goal_indices" in updates and isinstance(updates["goal_indices"], list):
            valid = []
            for idx in updates["goal_indices"]:
                if isinstance(idx, int) and 0 <= idx < len(GOAL_OPTIONS):
                    valid.append(GOAL_OPTIONS[idx])
            if valid:
                self.goals = valid[:10]

        # constraints: index list only
        if "constraint_indices" in updates and isinstance(updates["constraint_indices"], list):
            valid = []
            for idx in updates["constraint_indices"]:
                if isinstance(idx, int) and 0 <= idx < len(CONSTRAINT_OPTIONS):
                    valid.append(CONSTRAINT_OPTIONS[idx])
            if valid:
                self.constraints = valid[:10]

        # recent_decisions: pass-through
        if "recent_decisions" in updates and isinstance(updates["recent_decisions"], list):
            self.recent_decisions = [str(x) for x in updates["recent_decisions"]][:10]

    @staticmethod
    def options_text() -> str:
        lines = ["memory_update field options:"]
        lines.append("- risk_tolerance (budget sensitivity): " + ", ".join(
            f"{i}={v}" for i, v in enumerate(BUDGET_ORDER)
        ))
        lines.append("- goal_indices (select all that apply):")
        lines.append("    " + ", ".join(f"{i}={v}" for i, v in enumerate(GOAL_OPTIONS)))
        lines.append("- constraint_indices (select all that apply):")
        lines.append("    " + ", ".join(f"{i}={v}" for i, v in enumerate(CONSTRAINT_OPTIONS)))
        return "\n".join(lines)


def revealed_budget_from_choices(
    steps: List["RetailStep"],
    product_risk_db: Dict[str, int],
    n_steps: int = 5,
) -> str:
    """Infer budget sensitivity from actual user choices in the first *n_steps*.

    Looks up the price tier of each ``user_choice`` in *product_risk_db* and
    averages them.  The mean is mapped to an ordinal:

    * mean <= 2.0 → ``"tight"``
    * mean <= 3.5 → ``"moderate"``
    * else        → ``"flexible"``

    Mirrors :func:`revealed_risk_from_choices` in the finance domain.
    """
    scores: List[int] = []
    for step in steps[:n_steps]:
        choice = step.user_choice
        if isinstance(choice, str) and choice in product_risk_db:
            scores.append(product_risk_db[choice])
    if not scores:
        return "moderate"
    mean_score = sum(scores) / len(scores)
    if mean_score <= 2.0:
        return "tight"
    if mean_score <= 3.5:
        return "moderate"
    return "flexible"
