"""Conv-FinRe dataset loader and parser.

Loads the TheFinAI/conv-finre dataset from HuggingFace and provides
structured access to market snapshots, conversation messages, and
ground-truth labels for each user and step.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MarketSnapshot:
    """Parsed market data for a single ticker at a given step."""

    ticker: str
    ret_7d: float
    vol: float
    mdd: float
    mu: float
    current_price: float


@dataclass
class ConvFinReStep:
    """A single decision step for a user."""

    user_id: str
    step: int
    date: str
    messages: List[Dict[str, str]]
    market_snapshots: List[MarketSnapshot]
    labels: Dict[str, object]


@dataclass
class ConvFinReUser:
    """All data for a single user across their session."""

    user_id: str
    onboarding_messages: List[Dict[str, str]]
    steps: List[ConvFinReStep]


# Regex for parsing a single ticker line from the prompt field.
# Example: AMZN: 7d_ret=3.90%, vol=0.0117, mdd=0.0082015, mu=0.00646406,
#          var=0.000136391, current_price=$230.98,
#          price_trend=['$222.31', '$223.13', ...]
# Float pattern that handles scientific notation (e.g. 2.16504e-05)
_FLOAT = r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?"

_TICKER_LINE_RE = re.compile(
    r"^(?P<ticker>[A-Z]{1,5}):\s+"
    rf"7d_ret=(?P<ret>{_FLOAT})%,\s+"
    rf"vol=(?P<vol>{_FLOAT}),\s+"
    rf"mdd=(?P<mdd>{_FLOAT}),\s+"
    rf"mu=(?P<mu>{_FLOAT}),\s+"
    rf"var=(?P<var>{_FLOAT}),\s+"
    rf"current_price=\$(?P<price>{_FLOAT}),\s+"
    r"price_trend=\[(?P<trend>[^\]]+)\]",
    re.MULTILINE,
)

def _parse_market_snapshot(ticker_match: re.Match) -> MarketSnapshot:
    """Build a MarketSnapshot from a regex match on a ticker line."""
    return MarketSnapshot(
        ticker=ticker_match.group("ticker"),
        ret_7d=float(ticker_match.group("ret")),
        vol=float(ticker_match.group("vol")),
        mdd=float(ticker_match.group("mdd")),
        mu=float(ticker_match.group("mu")),
        current_price=float(ticker_match.group("price")),
    )


def _extract_onboarding(
    messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split messages into onboarding and panel phases.

    The system message containing "onboarding is complete" marks the boundary.
    Returns (onboarding_messages, panel_messages).
    """
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if msg.get("role") == "system" and "onboarding" in content.lower():
            return messages[:i], messages[i + 1 :]
    # No system boundary found — treat all as onboarding
    return list(messages), []


class ConvFinReDataset:
    """Loader for the TheFinAI/conv-finre HuggingFace dataset."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._cache_dir = cache_dir
        self._users: Dict[str, ConvFinReUser] = {}

    def load(self) -> "ConvFinReDataset":
        """Download/cache the dataset and parse all rows."""
        from datasets import load_dataset

        ds = load_dataset("TheFinAI/conv-finre", split="test", cache_dir=self._cache_dir)

        # Group rows by user_id
        rows_by_user: Dict[str, list] = {}
        for row in ds:
            uid = row["user_id"]
            rows_by_user.setdefault(uid, []).append(row)

        for uid, rows in rows_by_user.items():
            rows.sort(key=lambda r: r["step"])
            steps: List[ConvFinReStep] = []
            onboarding_msgs: List[Dict[str, str]] = []

            for row in rows:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in row["messages"]
                ]
                prompt = row["prompt"]

                # Parse market snapshots from prompt
                snapshots = [
                    _parse_market_snapshot(m)
                    for m in _TICKER_LINE_RE.finditer(prompt)
                ]
                # Extract labels
                labels = dict(row["labels"]) if row.get("labels") else {}

                step = ConvFinReStep(
                    user_id=uid,
                    step=row["step"],
                    date=row.get("date", ""),
                    messages=messages,
                    market_snapshots=snapshots,
                    labels=labels,
                )
                steps.append(step)

                # Extract onboarding from the first step
                if row["step"] == 1:
                    onboarding_msgs, _ = _extract_onboarding(messages)

            self._users[uid] = ConvFinReUser(
                user_id=uid,
                onboarding_messages=onboarding_msgs,
                steps=steps,
            )

        return self

    def users(self) -> List[str]:
        """Return sorted list of user IDs."""
        return sorted(self._users.keys())

    def get_user(self, user_id: str) -> ConvFinReUser:
        """Return data for a specific user."""
        return self._users[user_id]

    def __len__(self) -> int:
        return len(self._users)
