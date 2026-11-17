import tempfile, unittest
from pathlib import Path
from haven import Haven

class HavenTests(unittest.TestCase):
    def test_learning_and_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/"state.json")
            h=Haven(p)
            for _ in range(60): h.teach("A","C")
            prediction,probs=h.predict("A")
            self.assertEqual(prediction,"C")
            self.assertGreater(probs["C"],probs["B"])
            loaded=Haven(p)
            self.assertEqual(loaded.predict("A")[0],"C")
            self.assertEqual(loaded.steps,60)
    def test_new_node(self):
        with tempfile.TemporaryDirectory() as d:
            h=Haven(str(Path(d)/"state.json"))
            h.teach("NEW","A")
            self.assertIn("NEW",h.nodes)
    def test_reversal(self):
        with tempfile.TemporaryDirectory() as d:
            h=Haven(str(Path(d)/"state.json"))
            for _ in range(60):h.teach("A","B")
            self.assertEqual(h.predict("A")[0],"B")
            for _ in range(90):h.teach("A","D")
            self.assertEqual(h.predict("A")[0],"D")

if __name__=="__main__":unittest.main()
