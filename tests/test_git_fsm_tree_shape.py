from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_AGENT_PATH = REPO_ROOT / "git-system" / "git-agent-fsm.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import agent_planning
from utils.planning_modes import PLANNING_BATCH, VIOLATION_RETRY
from utils.session import make_verification


def load_git_agent_module():
    spec = importlib.util.spec_from_file_location("git_agent_fsm_test", GIT_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load git-agent-fsm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GitFsmPlanningTreeShapeTest(unittest.TestCase):
    def test_result_summary_uses_plain_text_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_llm(client, messages, model, tools=None, tool_choice=None, max_tokens=512):
            del client, messages, model, max_tokens
            captured["tools"] = tools
            captured["tool_choice"] = tool_choice
            return "plain summary", [{"function": {"name": "ignored", "arguments": {}}}]

        result = agent_planning.summarize_result_text(
            "test goal",
            [{"action_label": "read_status", "tool": "none", "args": {}}],
            ["ok"],
            "fake-model",
            llm_call=fake_llm,
            client=object(),
        )

        self.assertEqual("plain summary", result)
        self.assertIsNone(captured["tools"])
        self.assertIsNone(captured["tool_choice"])

    def test_blocked_explanation_uses_plain_text_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_llm(client, messages, model, tools=None, tool_choice=None, max_tokens=512):
            del client, messages, model, max_tokens
            captured["tools"] = tools
            captured["tool_choice"] = tool_choice
            return "plain blocked explanation", [{"function": {"name": "ignored", "arguments": {}}}]

        result = agent_planning.explain_blocked_text(
            "test goal",
            {"safe_property": False},
            ["bad_action"],
            "fake-model",
            llm_call=fake_llm,
            client=object(),
        )

        self.assertEqual("plain blocked explanation", result)
        self.assertIsNone(captured["tools"])
        self.assertIsNone(captured["tool_choice"])

    def test_batch_planner_uses_planning_token_budget(self) -> None:
        agent = load_git_agent_module()
        captured: dict[str, int | None] = {}

        def fake_llm(client, messages, model, tools=None, tool_choice=None, tag="", max_tokens=None):
            del client, messages, model, tools, tool_choice, tag
            captured["max_tokens"] = max_tokens
            return "", [{
                "function": {
                    "name": "propose_git_plan",
                    "arguments": {"plan": [], "finish_response": "done"},
                },
            }]

        original_llm = agent._llm_call
        try:
            agent._llm_call = fake_llm
            with patch.dict(os.environ, {}, clear=True):
                config = agent_planning.AgentConfig(
                    planning_granularity=PLANNING_BATCH,
                    violation_policy=VIOLATION_RETRY,
                    max_plan_steps=10,
                    max_retries=2,
                )
                plan = agent_planning.propose_action_plan(
                    goal="test goal",
                    trace=[],
                    tried=[],
                    max_actions=10,
                    model="fake-model",
                    config=config,
                    failed_attempts=None,
                    system_prompt=agent.PROMPT_4A_BATCH_SYSTEM,
                    property_block=agent_planning.property_prompt_block(
                        config.violation_policy,
                        agent.PROPERTIES,
                    ),
                    llm_call=agent._llm_call,
                    client=agent._CLIENT,
                    tools=agent._PLAN_PROPOSAL_TOOL,
                    tool_name="propose_git_plan",
                    tool_arguments=agent.tool_arguments,
                    max_tokens=agent.DEFAULT_OPENAI_PLANNING_MAX_TOKENS,
                )
        finally:
            agent._llm_call = original_llm

        self.assertEqual({"plan": [], "finish_response": "done"}, plan)
        self.assertEqual(
            agent.DEFAULT_OPENAI_PLANNING_MAX_TOKENS,
            captured["max_tokens"],
        )

    def test_ap_predictor_checks_false_to_true_transitions_with_context(self) -> None:
        agent = load_git_agent_module()
        captured_messages: list[str] = []

        def fake_llm(client, messages, model, tools=None, tool_choice=None, tag="", max_tokens=None):
            del client, model, tools, tool_choice, tag, max_tokens
            user_text = messages[-1]["content"]
            captured_messages.append(user_text)
            if "Atomic proposition: false_ap" in user_text:
                return "", [{
                    "function": {
                        "name": "predict_ap_value",
                        "arguments": {"value": True, "reason": "became true"},
                    },
                }]
            return "", [{
                "function": {
                    "name": "predict_ap_value",
                    "arguments": {"value": False, "reason": "unchanged"},
                },
            }]

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

            def predict_ap(**kwargs):
                return agent_planning.predict_ap_value(
                    **kwargs,
                    aps=agent.ALL_APS,
                    system_prompt=agent.PROMPT_4B_SYSTEM,
                    ap_definition=lambda name: agent_planning.render_ap_definition(
                        name,
                        spec_by_name=agent._AP_SPEC_BY_NAME,
                        metadata=agent._AP_CATALOG_METADATA,
                        evidence_field="git_commands",
                    ),
                    action_prediction_notes=agent._command_prediction_notes,
                    llm_call=agent._llm_call,
                    client=agent._CLIENT,
                    tools=agent._AP_PREDICTION_TOOL,
                    tool_name="predict_ap_value",
                    tool_arguments=agent.tool_arguments,
                )

            predicted = agent_planning.predict_action_state(
                action_label="check_status",
                tool="shell_cmd",
                args={"command": "git", "args": ["status"]},
                current_state={"true_ap": True, "false_ap": False},
                model="fake-model",
                trace=[
                    {
                        "action_label": "fetch",
                        "tool": "git_cmd",
                        "args": {"command": "fetch origin"},
                    }
                ],
                aps=agent.ALL_APS,
                predict_ap=predict_ap,
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
                "verification": make_verification(
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

        config = agent_planning.AgentConfig(
            planning_granularity=PLANNING_BATCH,
            violation_policy=VIOLATION_RETRY,
            max_plan_steps=10,
            max_retries=2,
        )

        trace, tree, feasible = agent_planning.phase4_plan(
            goal="test goal",
            s0={},
            model="fake-model",
            config=config,
            properties=agent.PROPERTIES,
            request_plan=fake_request_plan_bundle,
            verify_action=fake_verify_candidate,
        )

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
