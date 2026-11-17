import tempfile
import unittest
from pathlib import Path
from integrated import IntegratedHaven


class IntegratedTests(unittest.TestCase):
    def test_experience_changes_both_learning_systems(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h = IntegratedHaven(str(root / "graph.json"), str(root / "agent.json"))
            result = h.run(20)
            self.assertEqual(result["steps"], 20)
            self.assertGreater(result["graph_edges"], 0)
            self.assertTrue(h.agent.q)
            self.assertTrue((root / "graph.json").exists())
            self.assertTrue((root / "agent.json").exists())

    def test_experience_survives_reinitialization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            graph, agent = str(root / "graph.json"), str(root / "agent.json")
            h = IntegratedHaven(graph, agent)
            h.run(12)
            restored = IntegratedHaven(graph, agent)
            self.assertEqual(restored.agent.total_steps, 12)
            self.assertEqual(restored.graph.steps, 24)

    def test_outcome_graph_contains_actual_observation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h = IntegratedHaven(str(root / "graph.json"), str(root / "agent.json"))
            event = h.act()
            decision = h.action_symbol(event["state"], event["action"])
            outcome = h.symbol(event["next_state"])
            self.assertIn(h.graph.key(decision, outcome), h.graph.edges)


if __name__ == "__main__":
    unittest.main()
