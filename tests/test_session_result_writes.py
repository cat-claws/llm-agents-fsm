from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils import session


class SessionResultWriteTest(unittest.TestCase):
    def test_tree_html_renders_only_on_final_write(self) -> None:
        calls: list[Path] = []
        original_render = session.render_saved_tree_html
        session.render_saved_tree_html = lambda path: calls.append(Path(path))
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                record = session.make_session(
                    agent="test-agent",
                    model="test-model",
                    domain="test",
                    request="test request",
                )

                result_path = session.start_result_session(record, tmpdir)
                self.assertIsNotNone(result_path)
                self.assertEqual([], calls)

                session.checkpoint_result(record, result_path)
                self.assertEqual([], calls)

                session.write_result(record, tmpdir, result_path)
                self.assertEqual([Path(result_path)], calls)
        finally:
            session.render_saved_tree_html = original_render

    def test_checkpoint_preserves_planning_node_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            record = session.make_session(
                agent="test-agent",
                model="test-model",
                domain="test",
                request="test request",
            )
            node = session.make_planning_node(
                node_id=0,
                parent_node_id=None,
                depth=0,
                state_before={},
            )
            session.append_node(record["planning_tree"], node)

            result_path = session.start_result_session(record, tmpdir)
            self.assertIs(record["planning_tree"]["nodes"][0], node)

            session.set_node_outcome(
                node,
                result="accepted",
                action_label="move_grasper",
                tool="simulator_action",
                args={"x": 0.1, "y": -0.15},
                state_after={"grasper_lowered": False},
            )
            session.checkpoint_result(record, result_path)

            tree_node = record["planning_tree"]["nodes"][0]
            self.assertIs(tree_node, node)
            self.assertEqual("accepted", tree_node["result"])
            self.assertEqual("move_grasper", tree_node["action"]["label"])


if __name__ == "__main__":
    unittest.main()
