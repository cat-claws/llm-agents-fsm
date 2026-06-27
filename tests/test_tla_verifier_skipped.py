from __future__ import annotations

import unittest

from utils import tla_verifier


class TlaVerifierSkippedTest(unittest.TestCase):
    def test_skipped_tlc_is_not_passed(self) -> None:
        original_find_jar = tla_verifier._find_jar
        tla_verifier._find_jar = lambda: None
        try:
            result = tla_verifier.verify_fsm_trace(
                {"ap": False},
                ["act"],
                [{"ap": True}],
                ["ap"],
                [],
            )
        finally:
            tla_verifier._find_jar = original_find_jar

        self.assertFalse(result["passed"])
        self.assertTrue(result["tlc_result"]["skipped"])
        self.assertIn("tla2tools.jar not found", result["tlc_result"]["reason"])


if __name__ == "__main__":
    unittest.main()
