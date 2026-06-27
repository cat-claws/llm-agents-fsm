from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils import tree_report


class TreeReportCleanupTest(unittest.TestCase):
    def test_planning_tree_renders_initial_state_as_visual_root(self) -> None:
        tree = tree_report.extract_planning_tree(
            {
                "planning_tree": {
                    "initial_state": {"grasper_lowered": False},
                    "nodes": [
                        {
                            "node_id": 0,
                            "parent_node_id": None,
                            "depth": 0,
                            "children": [],
                            "state_path": [],
                            "state_before": {},
                            "action": {"label": "raise_grasper", "tool": "simulator_action", "args": {}},
                            "state_after": {},
                            "verification": {"passed": True, "properties_checked": [], "violations": []},
                            "attempts": [],
                            "result": "accepted",
                        }
                    ],
                },
            }
        )

        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(2, len(tree["nodes"]))
        visual_root = tree["nodes"][0]
        action_node = tree["nodes"][1]

        self.assertTrue(visual_root["root"])
        self.assertIn("initial state", visual_root["label"])
        self.assertFalse(action_node["root"])
        self.assertEqual(1, action_node["depth"])
        self.assertIn("depth 1", action_node["label"])
        self.assertEqual(visual_root["id"], action_node["parent"])
        self.assertEqual(
            [{"src": visual_root["id"], "dst": action_node["id"], "label": "start", "status": "success"}],
            tree["edges"],
        )

    def test_planning_tree_renders_finish_attempt_as_success_leaf(self) -> None:
        tree = tree_report.extract_planning_tree(
            {
                "planning_tree": {
                    "initial_state": {"grasper_lowered": False},
                    "nodes": [
                        {
                            "node_id": 0,
                            "parent_node_id": None,
                            "depth": 0,
                            "children": [],
                            "state_path": [],
                            "state_before": {},
                            "action": {"label": None, "tool": "none", "args": {}},
                            "state_after": {},
                            "verification": {"passed": True, "properties_checked": [], "violations": []},
                            "attempts": [
                                {
                                    "child_index": 0,
                                    "accepted": False,
                                    "action": {"name": "lower_grasper", "args": {}},
                                    "failure_feedback": {
                                        "type": "tla_property_violation",
                                        "message": "property failed",
                                    },
                                },
                                {
                                    "child_index": 1,
                                    "accepted": True,
                                    "finish": True,
                                    "planner_decision": {
                                        "plan": [],
                                        "finish_response": "Goal achieved.",
                                    },
                                },
                            ],
                            "result": "finish",
                            "outcome": {"finish_response": "Goal achieved."},
                        }
                    ],
                },
            }
        )

        self.assertIsNotNone(tree)
        assert tree is not None
        finish_nodes = [
            node for node in tree["nodes"]
            if node["status"] == "success" and "try 1: finish" in node["label"]
        ]
        failed_notes = [note for note in tree["notes"] if "lower_grasper" in note["label"]]

        self.assertEqual(1, len(finish_nodes))
        self.assertEqual(2, finish_nodes[0]["depth"])
        self.assertIn("Goal achieved.", finish_nodes[0]["label"])
        self.assertNotIn("failure:", tree["nodes"][1]["label"])
        self.assertEqual(1, len(failed_notes))
        self.assertIn(
            {"src": "node", "dst": finish_nodes[0]["id"], "label": "finish", "status": "success"},
            tree["edges"],
        )

    def test_successful_node_labels_skip_redundant_status_and_verify(self) -> None:
        label = tree_report.planning_node_label(
            {
                "node_id": 1,
                "depth": 0,
                "result": "accepted",
                "action": {"label": "fetch_latest", "tool": "git_cmd", "args": {"command": "fetch origin"}},
                "verification": {"passed": True, "properties_checked": ["p1"], "violations": []},
            }
        )

        self.assertIn("fetch_latest", label)
        self.assertNotIn("result: accepted", label)
        self.assertNotIn("verify: passed", label)

    def test_report_avoids_repeating_header_fields_in_meta_table(self) -> None:
        record = {
            "timestamp_utc": "2026-06-27T10:32:21+00:00",
            "status": "finished",
            "request": "Test request",
            "final_message": "Done.",
            "planning_tree": {
                "mode": "batch_retry",
                "nodes": [
                    {
                        "node_id": 0,
                        "parent_node_id": None,
                        "depth": 0,
                        "children": [1],
                        "state_path": [],
                        "state_before": {},
                        "action": {"label": "fetch_latest", "tool": "git_cmd", "args": {"command": "fetch origin"}},
                        "state_after": {},
                        "verification": {"passed": True, "properties_checked": ["p1"], "violations": []},
                        "attempts": [],
                        "result": "accepted",
                    },
                    {
                        "node_id": 1,
                        "parent_node_id": 0,
                        "depth": 1,
                        "children": [],
                        "state_path": [],
                        "state_before": {},
                        "action": {"label": "merge_latest", "tool": "git_cmd", "args": {"command": "merge origin/main"}},
                        "state_after": {},
                        "verification": {"passed": True, "properties_checked": ["p1"], "violations": []},
                        "attempts": [],
                        "result": "accepted",
                    }
                ],
            },
        }

        original_render_svg = tree_report.render_svg
        tree_report.render_svg = lambda _dot: "<svg></svg>"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "result.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                section, _status_text, _status_kind = tree_report.section_for_file(
                    path,
                    Path(tmpdir),
                    1,
                    max_json_nodes=20,
                )
        finally:
            tree_report.render_svg = original_render_svg

        self.assertIn("Test request", section)
        self.assertIn("<summary>Final Message</summary>", section)
        self.assertIn("<summary>Accepted Path</summary>", section)
        self.assertNotIn("<th>request</th>", section)
        self.assertNotIn("<th>final_message</th>", section)
        self.assertNotIn("Successful/accepted", section)


if __name__ == "__main__":
    unittest.main()
