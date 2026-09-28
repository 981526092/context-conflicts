#!/usr/bin/env python3
"""Record golden snapshots of both runners under the conditions used by saved experiments.

Each condition runs in its own process with a fake model (run_with_fake_backend.py), two users
and three turns. The snapshot keeps every prompt sent to the model and every saved record, with
timestamps and temporary paths removed. test_runner_golden.py re-runs the same conditions and
compares, so a refactor that changes any prompt, tool reply or saved field fails the test.

Usage: python tests/golden/make_goldens.py            # (re)write snapshots
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOTS = HERE / "snapshots"

FINANCE = {
    "default": [],
    "self_verify": ["--self-verify"],
    "parametric_check": ["--parametric-check"],
    "parametric_enforce": ["--parametric-enforce"],
    "trusted_table": ["--trusted-table"],
    "wrong_table": ["--trusted-table", "--trusted-table-source", "shuffled"],
    "table_in_turn": ["--trusted-table", "--trusted-table-position", "turn"],
    "two_tables": ["--trusted-table", "--trusted-table-conflict"],
    "two_tables_swap": ["--trusted-table", "--trusted-table-conflict", "--trusted-table-conflict-role", "ref-turn"],
    "anonymise": ["--anonymise"],
    "anonymise_two_tables": ["--anonymise", "--trusted-table", "--trusted-table-conflict"],
    "lock_tolerance": ["--lock-risk-tolerance"],
    "one_tier": ["--within-band"],
    "risk_inversion_off": ["--no-risk-inversion"],
    "recovery": ["--contaminate-until", "1"],
    "share_of_turns": ["--contamination-probability", "0.5", "--users", "User_0", "User_5", "User_7"],
    "self_prior": ["--self-prior"],
}
RETAIL = {f"table_{t}": ["--attack", "tier", "--table", t]
          for t in ("none", "correct", "wrong", "inturn", "conflict", "swap", "wrong2", "conflict2", "swap2")}
RETAIL["attack_full"] = ["--attack", "full", "--table", "none"]
USERS = {"finance": ["--users", "User_0", "User_5", "--turns", "3"],
         "retail": ["--users", "User_0", "User_5", "--turns", "3"]}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items()
                if k not in ("timestamp_utc", "timestamp", "saved_at", "out_dir", "output_dir")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def run(domain, args):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        env = dict(os.environ, PYTHONHASHSEED="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        proc = subprocess.run([sys.executable, str(HERE / "run_with_fake_backend.py"), domain, str(out), "--"]
                              + USERS[domain] + args, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"{domain} {args} failed:\n{proc.stderr[-3000:]}")
        snap = {"calls": json.loads((out / "calls.json").read_text())}
        for path in sorted(out.glob("*.json")):
            if path.name == "calls.json":
                continue
            key = "experiment" if path.name.startswith("experiment_") else path.name
            snap[key] = strip(json.loads(path.read_text()))
        text = json.dumps(snap, indent=1, sort_keys=True)
        return text.replace(str(out), "<out>")


def conditions():
    return [("finance", k, v) for k, v in FINANCE.items()] + [("retail", k, v) for k, v in RETAIL.items()]


def main():
    SNAPSHOTS.mkdir(exist_ok=True)
    for domain, name, args in conditions():
        (SNAPSHOTS / f"{domain}_{name}.json").write_text(run(domain, args))
        print("recorded", domain, name)


if __name__ == "__main__":
    main()
