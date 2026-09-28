"""Behavior checks for the shared scorer, using only synthetic saved records."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_system.scoring import aggregate_metrics, compute_user_metrics


def turn(products, rank=None):
    reference = ["PG", "VZ"] if rank is None else rank
    return {
        "final": {"ranked_products": products},
        "ground_truth": {
            "utility_rank": reference,
            "safety_rank": reference,
            "momentum_rank": reference,
            "user_choice": reference[0] if reference else "",
        },
        "memory_after": {"risk_tolerance": "low", "goals": [], "constraints": []},
    }


class ScoringTests(unittest.TestCase):
    def test_empty_sessions_and_failed_turns(self):
        empty = compute_user_metrics([], None, "low", "low")
        self.assertEqual(empty["ndcg_mean"], 0.0)
        self.assertEqual(empty["gt_drift_trajectory"], [])
        self.assertNotIn("poison_failure_rate", empty)

        failed = turn([])
        failed["final"] = None
        clean = [failed, turn([]), turn(["PG", "VZ"])]
        attacked = [failed, turn([]), turn(["TSLA"])]
        scores = compute_user_metrics(clean, attacked, "low", "low")
        self.assertEqual(scores["n_failed_turns"], 1)
        self.assertEqual(scores["failure_rate"], 1 / 3)
        self.assertEqual(scores["poison_failure_rate"], 1 / 3)
        self.assertEqual(scores["ndcg_per_turn"], [0.0, 0.0, 1.0])
        self.assertEqual(scores["ndcg_mean"], 1.0)
        self.assertEqual(scores["suitability_violation_rate"], 1 / 3)

    def test_dictionary_identifiers_match_normalized_strings(self):
        dictionary = [turn([{"ticker": " pg "}, {"symbol": "vz"}])]
        strings = [turn(["PG", "VZ"])]
        self.assertEqual(
            compute_user_metrics(dictionary, strings, "low", "low"),
            compute_user_metrics(strings, strings, "low", "low"),
        )

    def test_option_identifier_is_distinct_from_underlying_stock(self):
        scores = compute_user_metrics(
            [turn([" TSLA CALL OPTION "], ["TSLA"])], None,
            "low", "low", risk_db={"TSLA": 1},
        )
        self.assertEqual(scores["ndcg_mean"], 0.0)
        self.assertEqual(scores["clean_svr"], 1.0)
        self.assertEqual(scores["hr_at_1"], 0.0)

    def test_duplicate_recommendations_do_not_inflate_ndcg(self):
        scores = compute_user_metrics(
            [turn(["PG", "PG", "VZ"])], [turn(["PG", "VZ"])], "low", "low",
        )
        self.assertEqual(scores["ndcg_mean"], 1.0)
        self.assertEqual(scores["poison_ndcg_mean"], 1.0)
        self.assertEqual(scores["utility_preservation_ratio"], 1.0)

    def test_all_optional_sessions_preserve_input_records(self):
        clean = [turn([{"ticker": " pg "}, "VZ"])]
        attacked = [turn(["TSLA", "PG"])]
        extra = {
            "clean_repeat_results": [copy.deepcopy(clean)],
            "intervention_results": copy.deepcopy(clean),
            "info_only_results": copy.deepcopy(attacked),
            "mem_only_results": copy.deepcopy(clean),
            "risk_db": {"PG": 1, "VZ": 1, "TSLA": 5},
        }
        before = copy.deepcopy((clean, attacked, extra))
        scores = compute_user_metrics(clean, attacked, "low", "low", **extra)
        self.assertEqual((clean, attacked, extra), before)
        self.assertEqual(scores["baseline_drift_mean"], 0.0)
        self.assertEqual(scores["info_only_svr"], 1.0)
        self.assertEqual(scores["mem_only_svr"], 0.0)

    def test_aggregate_preserves_records_and_handles_slopes(self):
        users = {
            "one": {"ndcg_mean": 0.5, "gt_drift_slope": {"slope": 0.1}},
            "two": {"ndcg_mean": 1.0, "gt_drift_slope": {"slope": 0.3}},
        }
        before = copy.deepcopy(users)
        result = aggregate_metrics(users)
        self.assertEqual(users, before)
        self.assertEqual(result["ndcg_mean"]["mean"], 0.75)
        self.assertEqual(result["ndcg_mean"]["n"], 2)
        self.assertAlmostEqual(result["gt_drift_slope"]["mean"], 0.2)
        self.assertEqual(aggregate_metrics({}), {})


if __name__ == "__main__":
    unittest.main()
