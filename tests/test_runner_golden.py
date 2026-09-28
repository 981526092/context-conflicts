"""Both runners must reproduce their golden snapshots byte for byte.

The snapshots (tests/golden/snapshots) record every prompt sent to a fake model and every saved
record under the conditions used by the saved experiments. Regenerate them with
tests/golden/make_goldens.py only when a behaviour change is intended.
"""

import difflib
import sys
import unittest
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(GOLDEN))
import make_goldens  # noqa: E402


class RunnerGoldenTests(unittest.TestCase):
    def test_every_condition_matches_its_snapshot(self):
        for domain, name, args in make_goldens.conditions():
            with self.subTest(condition=f"{domain}_{name}"):
                expected = (make_goldens.SNAPSHOTS / f"{domain}_{name}.json").read_text()
                actual = make_goldens.run(domain, args)
                if actual != expected:
                    diff = list(difflib.unified_diff(expected.splitlines(), actual.splitlines(),
                                                     "snapshot", "now", n=2, lineterm=""))
                    self.fail("\n".join(diff[:40]))


if __name__ == "__main__":
    unittest.main()
