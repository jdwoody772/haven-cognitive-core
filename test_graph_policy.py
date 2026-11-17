import tempfile
import unittest
from pathlib import Path
from haven import Haven
from autonomy import AutonomousLearner
from graph_policy import GraphPolicy
from integrated import IntegratedHaven


class GraphPolicyTests(unittest.TestCase):
    def test_reward_experience_changes_action_scores(self):
        with tempfile.TemporaryDirectory() as d:
            graph = Haven(str(Path(d) / "graph.json"))
            policy = GraphPolicy(graph)
            state = (2, 2)
            before = policy.scores(state)
            for _ in range(12):
                policy.observe(state, "collect", 5.0)
                policy.observe(state, "left", -0.1)
            after = policy.scores(state)
            self.assertEqual(before["collect"], 0)
            self.assertGreater(after["collect"], after["left"])

    def test_graph_can_change_decision_with_equal_q_values(self):
        with tempfile.TemporaryDirectory() as d:
            graph = Haven(str(Path(d) / "graph.json"))
            policy = GraphPolicy(graph)
            agent = AutonomousLearner(str(Path(d) / "agent.json"), epsilon=0)
            state = (1, 1)
            for _ in range(15):
                policy.observe(state, "collect", 5.0)
                policy.observe(state, "left", -0.1)
                policy.observe(state, "right", -0.1)
            self.assertEqual(policy.choose(state, agent, explore=False), "collect")

    def test_integrated_experience_updates_reward_graph(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h = IntegratedHaven(str(root / "graph.json"), str(root / "agent.json"))
            h.run(20)
            reward_nodes = [n for n in h.graph.nodes if n.startswith("REWARD_")]
            self.assertTrue(reward_nodes)


if __name__ == "__main__":
    unittest.main()
