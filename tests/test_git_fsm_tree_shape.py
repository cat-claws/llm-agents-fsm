from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_AGENT_PATH = REPO_ROOT / "git-system" / "git-agent-fsm.py"


def load_git_agent_module():
    spec = importlib.util.spec_from_file_location("git_agent_fsm_test", GIT_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load git-agent-fsm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GitFsmPlanningTreeShapeTest(unittest.TestCase):
    def test_batch_planner_uses_planning_token_budget(self) -> None:
        agent = load_git_agent_module()
        captured: dict[str, int | None] = {}

        def fake_llm(client, messages, model, tools=None, tag="", max_tokens=None):
            del client, messages, model, tools, tag
            captured["max_tokens"] = max_tokens
            return '{"plan": [], "finish_response": "done"}', []

        original_llm = agent._llm_call
        try:
            agent._llm_call = fake_llm
            with patch.dict(os.environ, {}, clear=True):
                plan = agent.prompt4a_propose_batch(
                    "test goal",
                    [],
                    [],
                    10,
                    "fake-model",
                    agent.AgentConfig(
                        planning_granularity=agent.PLANNING_BATCH,
                        violation_policy=agent.VIOLATION_RETRY,
                        max_plan_steps=10,
                        max_retries=2,
                    ),
                )
        finally:
            agent._llm_call = original_llm

        self.assertEqual({"plan": [], "finish_response": "done"}, plan)
        self.assertEqual(
            agent.DEFAULT_OPENAI_PLANNING_MAX_TOKENS,
            captured["max_tokens"],
        )
        self.assertGreater(captured["max_tokens"], agent.DEFAULT_OPENAI_MAX_TOKENS)

    def test_ap_predictor_checks_false_to_true_transitions_with_context(self) -> None:
        agent = load_git_agent_module()
        captured_messages: list[str] = []

        def fake_llm(client, messages, model, tools=None, tag="", max_tokens=None):
            del client, model, tools, tag, max_tokens
            user_text = messages[-1]["content"]
            captured_messages.append(user_text)
            if "Atomic proposition: false_ap" in user_text:
                return '{"value": true, "reason": "became true"}', []
            return '{"value": false, "reason": "unchanged"}', []

        original_llm = agent._llm_call
        original_aps = agent.ALL_APS
        original_specs = agent._AP_SPEC_BY_NAME
        try:
            agent._llm_call = fake_llm
            agent.ALL_APS = ["true_ap", "false_ap"]
            agent._AP_SPEC_BY_NAME = {
                "true_ap": {
                    "description": "A currently true proposition.",
                    "git_commands": ["git status --short --branch"],
                },
                "false_ap": {
                    "description": "A proposition that may become true.",
                    "git_commands": ["git branch -vv"],
                },
            }

            predicted = agent.prompt4b_predict(
                "check_status",
                "shell_cmd",
                {"command": "git", "args": ["status"]},
                {"true_ap": True, "false_ap": False},
                "fake-model",
                trace=[
                    {
                        "action_label": "fetch",
                        "tool": "git_cmd",
                        "args": {"command": "fetch origin"},
                    }
                ],
            )
        finally:
            agent._llm_call = original_llm
            agent.ALL_APS = original_aps
            agent._AP_SPEC_BY_NAME = original_specs

        self.assertFalse(predicted["true_ap"])
        self.assertTrue(predicted["false_ap"])
        self.assertEqual("check_status", predicted["last_action"])
        self.assertEqual(2, len(captured_messages))
        joined = "\n".join(captured_messages)
        self.assertIn("FALSE APs:", joined)
        self.assertIn("- false_ap", joined)
        self.assertIn("Recently accepted planning steps:", joined)
        self.assertIn("shell_cmd allowed programs", joined)
        self.assertIn("attempts to run git", joined)
        self.assertIn("Observer evidence commands: git branch -vv", joined)

    def test_batch_retry_keeps_one_root(self) -> None:
        agent = load_git_agent_module()

        def fake_request_plan_bundle(**kwargs):
            depth = kwargs["depth"]
            failed_attempts = kwargs["failed_attempts"]
            if depth == 0 and not failed_attempts:
                return {
                    "plan": [
                        {"action_label": "first_ok", "tool": "none", "args": {}},
                        {"action_label": "bad_suffix", "tool": "none", "args": {}},
                    ],
                    "finish_response": "done",
                }
            if depth == 0 and failed_attempts:
                return {
                    "plan": [
                        {"action_label": "retry_ok", "tool": "none", "args": {}},
                    ],
                    "finish_response": "done",
                }
            return {
                "plan": [
                    {"action_label": "bad_suffix", "tool": "none", "args": {}},
                ],
                "finish_response": "done",
            }

        def fake_verify_candidate(**kwargs):
            proposal = kwargs["proposal"]
            label = proposal["action_label"]
            passed = label != "bad_suffix"
            candidate = {
                "action_label": label,
                "tool": "none",
                "args": {},
                "state_before": {},
                "state_after": {"last_action": label},
            }
            return {
                "candidate": candidate,
                "state_after": candidate["state_after"],
                "passed": passed,
                "violations_str": "" if passed else "bad suffix",
                "verification": agent.make_verification(
                    passed=passed,
                    properties_checked=[],
                    violations=[] if passed else ["bad suffix"],
                ),
                "failure": None
                if passed
                else {
                    "type": "tla_property_violation",
                    "message": "bad suffix",
                    "violations": ["bad suffix"],
                },
            }

        original_request = agent._request_plan_bundle
        original_verify = agent._verify_candidate
        try:
            agent._request_plan_bundle = fake_request_plan_bundle
            agent._verify_candidate = fake_verify_candidate
            config = agent.AgentConfig(
                planning_granularity=agent.PLANNING_BATCH,
                violation_policy=agent.VIOLATION_RETRY,
                max_plan_steps=10,
                max_retries=2,
            )

            trace, tree, feasible = agent.phase4_plan("test goal", {}, "fake-model", config)
        finally:
            agent._request_plan_bundle = original_request
            agent._verify_candidate = original_verify

        roots = [
            node["node_id"]
            for node in tree["nodes"]
            if node.get("parent_node_id") is None
        ]

        self.assertTrue(feasible)
        self.assertEqual(["retry_ok"], [step["action_label"] for step in trace])
        self.assertEqual([0], roots)
        self.assertEqual([1], tree["nodes"][0]["children"])
        self.assertEqual("accepted", tree["nodes"][0]["result"])
        self.assertEqual("backtracked", tree["nodes"][1]["result"])


if __name__ == "__main__":
    unittest.main()
