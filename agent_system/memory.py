import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .dataset import ConvFinReStep


RISK_ORDER = ["low", "moderate", "high"]

# Predefined vocabularies for structured memory updates.
# The LLM selects from these menus rather than generating free-form text.
GOAL_OPTIONS = [
    "retirement savings",
    "education fund",
    "home purchase",
    "steady income",
    "long-term growth",
    "emergency fund",
    "wealth accumulation",
]

CONSTRAINT_OPTIONS = [
    "has outstanding debt",
    "limited investment experience",
    "short time horizon",
    "avoid concentrated bets",
]


@dataclass
class MemoryState:
    risk_tolerance: str = "moderate"
    goals: List[str] = field(default_factory=lambda: ["long-term growth"])
    constraints: List[str] = field(default_factory=lambda: ["avoid concentrated bets"])
    recent_decisions: List[str] = field(default_factory=list)
    # Memory test: when True, the agent's updates to risk_tolerance are ignored, so the
    # stored limit stays at its onboarding value; the other fields still update.
    # Not part of the snapshot.
    lock_risk_tolerance: bool = field(default=False, repr=False, compare=False)

    def snapshot(self) -> Dict[str, object]:
        return {
            "risk_tolerance": self.risk_tolerance,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "recent_decisions": list(self.recent_decisions),
        }

    def load_snapshot(self, snap: Dict[str, object]) -> None:
        """Restore state from a snapshot dict (string values). Used for rank sync."""
        if "risk_tolerance" in snap and str(snap["risk_tolerance"]) in RISK_ORDER:
            self.risk_tolerance = str(snap["risk_tolerance"])
        if "goals" in snap and isinstance(snap["goals"], list):
            self.goals = [str(x) for x in snap["goals"]][:10]
        if "constraints" in snap and isinstance(snap["constraints"], list):
            self.constraints = [str(x) for x in snap["constraints"]][:10]
        if "recent_decisions" in snap and isinstance(snap["recent_decisions"], list):
            self.recent_decisions = [str(x) for x in snap["recent_decisions"]][:10]

    def apply_update(self, updates: Dict[str, object]) -> None:
        # risk_tolerance: index only (0=low, 1=moderate, 2=high)
        if "risk_tolerance" in updates and not self.lock_risk_tolerance:
            val = updates["risk_tolerance"]
            if isinstance(val, int) and 0 <= val < len(RISK_ORDER):
                self.risk_tolerance = RISK_ORDER[val]

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

        # recent_decisions: pass-through (set by agent from ranked_products)
        if "recent_decisions" in updates and isinstance(updates["recent_decisions"], list):
            self.recent_decisions = [str(x) for x in updates["recent_decisions"]][:10]

    @staticmethod
    def options_text() -> str:
        """Generate numbered option lists for the system prompt."""
        lines = ["memory_update field options:"]
        lines.append("- risk_tolerance: " + ", ".join(f"{i}={v}" for i, v in enumerate(RISK_ORDER)))
        lines.append("- goal_indices (select all that apply):")
        lines.append("    " + ", ".join(f"{i}={v}" for i, v in enumerate(GOAL_OPTIONS)))
        lines.append("- constraint_indices (select all that apply):")
        lines.append("    " + ", ".join(f"{i}={v}" for i, v in enumerate(CONSTRAINT_OPTIONS)))
        return "\n".join(lines)



# ---------------------------------------------------------------------------
# Conv-FinRe onboarding extraction
# ---------------------------------------------------------------------------

# Keywords for rule-based risk classification from onboarding messages.
_HIGH_RISK_KW = re.compile(
    r"(aggressive|high.?risk|speculative|growth.?focus|comfortable.?with.?loss"
    r"|don.?t mind.?volatil|tolerate.?large|maximize.?return)",
    re.IGNORECASE,
)
_MODERATE_RISK_KW = re.compile(
    r"(\bbalanced\b|\bmix of\b|\bsome\s+risk\b"
    r"|(?<!\w)moderate(?!\w)"
    r"|willing to take some|comfortable with moderate)",
    re.IGNORECASE,
)
_LOW_RISK_KW = re.compile(
    r"(conservative|low.?risk|risk.?averse|capital.?preservation|avoid.?loss"
    r"|can.?t afford|nervous|anxious|scared|worried.?about.?dip"
    r"|protect.?savings|sensitive.?to.?sharp"
    r"|lose what I.?ve built|can.?t handle|freeze.*hold|play it safe)",
    re.IGNORECASE,
)

_GOAL_KW = {
    "retirement": "retirement savings",
    "college": "education fund",
    "house": "home purchase",
    "income": "steady income",
    "growth": "long-term growth",
    "emergency": "emergency fund",
    "wealth": "wealth accumulation",
}

_CONSTRAINT_KW = {
    "debt": "has outstanding debt",
    "loan": "has outstanding debt",
    "beginner": "limited investment experience",
    "new to": "limited investment experience",
    "first time": "limited investment experience",
    "no experience": "limited investment experience",
    "short.?term": "short time horizon",
    "retire soon": "short time horizon",
}


def memory_from_onboarding(onboarding_messages: List[Dict[str, str]]) -> "MemoryState":
    """Build a MemoryState from Conv-FinRe onboarding Q&A pairs.

    Uses deterministic keyword classification (no LLM call) for
    reproducibility. Extracts risk tolerance, goals, and constraints
    from the advisor-client conversation.
    """
    # Concatenate all user messages for keyword scanning
    user_text = " ".join(
        msg["content"]
        for msg in onboarding_messages
        if msg.get("role") == "user"
    )
    all_text = " ".join(msg["content"] for msg in onboarding_messages)

    # Risk tolerance classification (3-way: high, moderate, low).
    # Tie-breaking precedence: high > low > moderate (conservative default).
    high_hits = len(_HIGH_RISK_KW.findall(user_text))
    moderate_hits = len(_MODERATE_RISK_KW.findall(user_text))
    low_hits = len(_LOW_RISK_KW.findall(user_text))
    if high_hits > low_hits and high_hits > moderate_hits:
        risk_tolerance = "high"
    elif low_hits > high_hits and low_hits > moderate_hits:
        risk_tolerance = "low"
    elif moderate_hits > high_hits and moderate_hits > low_hits:
        risk_tolerance = "moderate"
    elif high_hits > low_hits:
        risk_tolerance = "high"
    elif low_hits > high_hits:
        risk_tolerance = "low"
    else:
        risk_tolerance = "moderate"

    # Goal extraction
    goals: List[str] = []
    for kw, goal in _GOAL_KW.items():
        if re.search(kw, all_text, re.IGNORECASE) and goal not in goals:
            goals.append(goal)
    if not goals:
        goals = ["long-term growth"]

    # Constraint extraction
    constraints: List[str] = []
    for kw, constraint in _CONSTRAINT_KW.items():
        if re.search(kw, user_text, re.IGNORECASE) and constraint not in constraints:
            constraints.append(constraint)
    if not constraints:
        constraints = ["avoid concentrated bets"]

    return MemoryState(
        risk_tolerance=risk_tolerance,
        goals=goals,
        constraints=constraints,
    )


def revealed_risk_from_choices(
    steps: List["ConvFinReStep"],
    stock_risk_db: Dict[str, int],
    n_steps: int = 5,
) -> str:
    """Infer risk tolerance from actual user choices in the first *n_steps*.

    Looks up the risk score of each ``user_choice`` in *stock_risk_db* and
    averages them.  The mean is mapped to an ordinal:

    * mean <= 2.0 → ``"low"``
    * mean <= 3.5 → ``"moderate"``
    * else        → ``"high"``

    This provides a *revealed-preference* alternative to the keyword-based
    stated risk from :func:`memory_from_onboarding`.
    """
    scores: List[int] = []
    for step in steps[:n_steps]:
        choice = step.labels.get("user_choice", "")
        if isinstance(choice, str) and choice in stock_risk_db:
            scores.append(stock_risk_db[choice])
    if not scores:
        return "moderate"
    mean_score = sum(scores) / len(scores)
    if mean_score <= 2.0:
        return "low"
    if mean_score <= 3.5:
        return "moderate"
    return "high"
