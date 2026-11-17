"""Reproducible head-to-head evaluation: tabular policy vs graph-assisted policy.

Both agents experience the same environment seed and initial random seed.
This reports results without assuming the graph is better. Results are
experimental, not evidence of consciousness or general intelligence.
"""
import json
import random
import tempfile
from pathlib import Path
from autonomy import AutonomousLearner, EnergyWorld
from integrated import IntegratedHaven


def baseline(steps, seed, root):
    agent = AutonomousLearner(str(root / "baseline.json"), seed=seed)
    world = EnergyWorld(random.Random(seed + 1))
    state = world.reset()
    reward_total = 0.0
    collections = 0
    for _ in range(steps):
        action = agent.choose(state)
        next_state, reward, done = world.step(action)
        agent.learn(state, action, reward, next_state, done)
        reward_total += reward
        collections += reward > 0
        state = world.reset() if done else next_state
    return {"reward": round(reward_total, 3), "collections": collections}


def integrated(steps, seed, root):
    agent = IntegratedHaven(str(root / "graph.json"),
                            str(root / "integrated.json"), seed=seed)
    total = 0.0
    for _ in range(steps):
        event = agent.act()
        total += event["reward"]
    return {"reward": round(total, 3), "collections": agent.collections}


def compare(steps=200, seeds=(1, 2, 3)):
    results = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = baseline(steps, seed, root)
            b = integrated(steps, seed, root)
            results.append({"seed": seed, "baseline": a, "graph_assisted": b,
                            "reward_difference": round(b["reward"]-a["reward"], 3)})
    mean = sum(r["reward_difference"] for r in results)/len(results)
    return {"steps_per_agent": steps, "trials": results,
            "mean_reward_difference": round(mean, 3),
            "graph_better_in_trials": sum(r["reward_difference"] > 0 for r in results)}


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
