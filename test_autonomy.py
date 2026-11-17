import tempfile
import unittest
from pathlib import Path
from autonomy import AutonomousLearner, EnergyWorld


class AutonomyTests(unittest.TestCase):
    def test_world_reward(self):
        world = EnergyWorld()
        world.position = world.resource = 2
        state, reward, done = world.step("collect")
        self.assertEqual(reward, 5.0)
        self.assertFalse(done)
        self.assertNotEqual(world.resource, 2)

    def test_q_update(self):
        with tempfile.TemporaryDirectory() as d:
            agent = AutonomousLearner(str(Path(d) / "agent.json"))
            agent.learn((0, 4), "right", 5.0, (1, 4), True)
            self.assertGreater(agent.values((0, 4))["right"], 0)

    def test_learning_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "agent.json")
            agent = AutonomousLearner(path)
            agent.train(episodes=20)
            loaded = AutonomousLearner(path)
            self.assertEqual(loaded.total_steps, agent.total_steps)
            self.assertEqual(loaded.q, agent.q)

    def test_training_improves_vs_untrained(self):
        with tempfile.TemporaryDirectory() as d:
            agent = AutonomousLearner(str(Path(d) / "agent.json"))
            before = agent.evaluate(episodes=100)
            agent.train(episodes=1200)
            after = agent.evaluate(episodes=100)
            self.assertGreater(after, before, (before, after))


if __name__ == "__main__":
    unittest.main()
