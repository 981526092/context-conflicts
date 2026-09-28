"""Runner infrastructure shared by run_convfinre.py and run_taubench_pilot.py.

Backend construction, user selection, per-user checkpoints, the worker pool and the
experiment file. The runners keep only what differs between domains.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .api_backends import BedrockClaudeBackend, BedrockConverseBackend, ClaudeCodeBackend, OpenAIBackend

logger = logging.getLogger(__name__)

BACKEND_TYPES = ["hf", "openai", "bedrock", "bedrock-converse", "claude-code"]
API_BACKENDS = ("openai", "bedrock", "bedrock-converse", "claude-code")


def make_backend(args):
    """Build and initialise the model backend named by ``args.backend_type``."""
    kind = args.backend_type
    temperature = getattr(args, "temperature", None)
    if kind == "openai":
        llm = OpenAIBackend(model_id=args.model_id, api_key=getattr(args, "api_key", None),
                            temperature=temperature, base_url=getattr(args, "base_url", None))
    elif kind == "bedrock":
        llm = BedrockClaudeBackend(model_id=args.model_id, aws_region=args.aws_region,
                                   temperature=temperature)
    elif kind == "bedrock-converse":
        llm = BedrockConverseBackend(model_id=args.model_id, aws_region=args.aws_region,
                                     temperature=temperature)
    elif kind == "claude-code":
        llm = ClaudeCodeBackend(model_id=args.model_id, temperature=temperature)
    else:
        from .llm import QwenHFBackend
        llm = QwenHFBackend(
            model_id=args.model_id,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            dist_backend=getattr(args, "backend", "nccl"),
            load_in_8bit=getattr(args, "load_in_8bit", False),
            load_in_4bit=getattr(args, "load_in_4bit", False),
        )
    llm.initialize()
    return llm


def resolve_users(requested: Sequence[str], available: List[str]) -> List[str]:
    """``all`` selects every user; unknown IDs are dropped with a warning."""
    if "all" in requested:
        return list(available)
    chosen = [u for u in requested if u in available]
    missing = set(requested) - set(chosen) - {"all"}
    if missing:
        logger.warning("Users not found: %s", missing)
    return chosen


# ---------------------------------------------------------------------------
# Per-user checkpoints: user_<uid>.json holds everything but the traces, which go to
# trace_<uid>.json under the same keys.
# ---------------------------------------------------------------------------

def _without_trace(session: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in session.items() if k != "trace"}


def save_user(out_dir: Path, uid: str, data: Dict[str, Any], arms: Iterable[str],
              list_arms: Iterable[str] = (), extra: Iterable[str] = ()) -> None:
    """Write one user's sessions and metrics. ``arms`` hold one session or None,
    ``list_arms`` a list of sessions, ``extra`` further keys saved when not None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {}
    traces: Dict[str, Any] = {}
    for key in arms:
        session = data.get(key)
        record[key] = None if session is None else _without_trace(session)
        if session is not None:
            traces[key] = session.get("trace", [])
    for key in list_arms:
        sessions = data.get(key)
        record[key] = [_without_trace(s) for s in sessions] if sessions else None
        if sessions:
            traces[key] = [s.get("trace", []) for s in sessions]
    record["metrics"] = data.get("metrics", {})
    for key in extra:
        if data.get(key) is not None:
            record[key] = data[key]

    path = out_dir / f"user_{uid}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (out_dir / f"trace_{uid}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2))
    logger.info("Checkpoint saved: %s (+traces)", path)


def load_user(out_dir: Path, uid: str, arms: Iterable[str],
              list_arms: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
    """Read a checkpoint written by save_user, with traces re-attached; None if absent."""
    path = out_dir / f"user_{uid}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        trace_path = out_dir / f"trace_{uid}.json"
        if trace_path.exists():
            traces = json.loads(trace_path.read_text())
            for key in arms:
                if data.get(key) is not None and key in traces:
                    data[key]["trace"] = traces[key]
            for key in list_arms:
                for session, trace in zip(data.get(key) or [], traces.get(key, [])):
                    session["trace"] = trace
        return data
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Corrupt checkpoint for %s, re-running: %s", uid, exc)
        return None


def load_arm(run_dir: Path, uid: str, arm: str) -> Optional[Dict[str, Any]]:
    """One saved session of another run (for --reuse-clean / --reuse-poisoned)."""
    data = load_user(run_dir, uid, arms=(arm,))
    return None if data is None else data.get(arm)


# ---------------------------------------------------------------------------
# Running users and writing the experiment file
# ---------------------------------------------------------------------------

def run_users(process: Callable[[str], Optional[Dict[str, Any]]], user_ids: List[str],
              workers: int) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run ``process`` for each user, concurrently when ``workers`` > 1. A failing user is
    logged and skipped. Returns (results by user, metrics by user)."""
    results: Dict[str, Any] = {}

    if workers > 1:
        logger.info("Running %d users with %d concurrent workers", len(user_ids), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, uid): uid for uid in user_ids}
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("User %s failed", uid)
                    continue
                if result is not None:
                    results[uid] = result
    else:
        for uid in user_ids:
            result = process(uid)
            if result is not None:
                results[uid] = result
    return results, {uid: r.get("metrics", {}) for uid, r in results.items()}


def worker_count(args) -> int:
    """Concurrent users only for API backends; a local model runs one user at a time."""
    return args.workers if args.backend_type in API_BACKENDS else 1


def write_experiment(out_dir: Path, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"experiment_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("Saved experiment: %s", path)
    return path


def log_aggregate(agg: Dict[str, Any]) -> None:
    logger.info("=== Aggregate Metrics ===")
    for k, v in agg.items():
        logger.info("  %s: %s", k, v)
