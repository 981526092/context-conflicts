"""Ticker anonymisation at the model boundary (--anonymise).

Every message sent to the model has its tickers replaced by fixed pseudonyms, and
every model output has the pseudonyms mapped back, so memory, scoring and traces
keep the real tickers. Conv-FinRe names stocks only by ticker (company names occur
0 times), so replacing tickers removes what the model knows about each stock.
"""

import random
import re
from typing import Any, Dict, List

TICKERS = ["PG", "VZ", "LIN", "XOM", "JPM", "MRK", "AMZN", "SPG", "MMM", "TSLA", "TQQQ"]

# Letters are drawn at random so that their order carries no risk information.
_letters = random.Random(7).sample("ABCDEFGHJKL", len(TICKERS))
PSEUDONYM: Dict[str, str] = {t: f"STK-{c}" for t, c in zip(TICKERS, _letters)}
_REAL = {v.split("-")[1]: k for k, v in PSEUDONYM.items()}

_FORWARD = re.compile(r"\b(" + "|".join(sorted(TICKERS, key=len, reverse=True)) + r")\b")
_BACKWARD = re.compile(r"\bSTK[- ]?([" + "".join(_REAL) + r"])\b", re.IGNORECASE)


# Price levels identify stocks (a re-identification probe on raw replies recovered 8 of
# 10 tickers for Claude), so the anonymised runs also drop them. Returns, volatility,
# drawdown and mu stay. Seven of 230 replayed user messages quote a price in the
# user's own words; those are left verbatim.
_PRICE_FIELDS = [
    re.compile(r',\s*"current_price":\s*-?[\d.]+'),
    re.compile(r'"current_price":\s*-?[\d.]+,\s*'),
    re.compile(r',?\s*current_price=\$?-?[\d.]+'),
    re.compile(r',?\s*price_trend=\[[^\]]*\]'),
]


def anonymise(text: str) -> str:
    for pat in _PRICE_FIELDS:
        text = pat.sub("", text)
    return _FORWARD.sub(lambda m: PSEUDONYM[m.group(1)], text)


def deanonymise(text: str) -> str:
    return _BACKWARD.sub(lambda m: _REAL[m.group(1).upper()], text)


class AnonymisingBackend:
    """Wraps any backend with generate(messages, enable_thinking)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        masked = [{**m, "content": anonymise(m["content"])} if isinstance(m.get("content"), str) else m
                  for m in messages]
        return deanonymise(self._inner.generate(masked, enable_thinking=enable_thinking))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
