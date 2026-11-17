"""Autonomous action-learning laboratory.

A small, deliberately observable environment: an agent can move left/right,
collect energy, and learn the consequences of actions through reinforcement.
This is a benchmark, not a model of sentience.
"""
import json
import os
import random
from pathlib import Path


class EnergyWorld:
    """Five positions. Energy pickup changes location after collection."""
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.position = 0
        self.resource = 4
        self.energy = 10
        self.steps = 0

    def reset(self):
        self.position = self.rng.randrange(5)
        self.resource = self.rng.randrange(5)
        self.energy = 10
        self.steps = 0
        return self.observe()

    def observe(self):
        return (self.position, self.resource)

    def step(self, action):
        if action not in ("left", "right", "collect"):
            raise ValueError("invalid action")
        self.steps += 1
        self.energy -= 1
        reward = -0.1
        if action == "left":
            self.position = max(0, self.position - 1)
        elif action == "right":
            self.position = min(4, self.position + 1)
        elif self.position == self.resource:
            reward = 5.0
            self.energy += 5
            # Always move resource away from the collected position.
            self.resource = self.rng.choice([i for i in range(5) if i != self.position])
        if self.energy <= 0:
            reward -= 2.0
        return self.observe(), reward, self.energy <= 0


class AutonomousLearner:
    ACTIONS = ("left", "right", "collect")

    def __init__(self, path="agent_state.json", seed=13, epsilon=0.15):
        self.path = Path(path)
        self.rng = random.Random(seed)
        self.q = {}
        self.alpha = 0.25
        self.gamma = 0.9
        self.epsilon = epsilon
        self.total_steps = 0
        self.total_reward = 0.0
        self.recent = []
        if self.path.exists():
            self.load()

    @staticmethod
    def state_key(state):
        return "%d:%d" % state

    def values(self, state):
        key = self.state_key(state)
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in self.ACTIONS}
        return self.q[key]

    def choose(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.ACTIONS)
        values = self.values(state)
        top = max(values.values())
        best = [a for a, v in values.items() if v == top]
        return self.rng.choice(best)

    def learn(self, state, action, reward, next_state, terminal):
        current = self.values(state)
        future = 0 if terminal else max(self.values(next_state).values())
        error = reward + self.gamma * future - current[action]
        current[action] += self.alpha * error
        self.total_steps += 1
        self.total_reward += reward
        self.recent.append({"state": list(state), "action": action,
                            "reward": reward, "error": round(error, 4)})
        self.recent = self.recent[-100:]
        return error

    def train(self, episodes=500, max_steps=50, seed=17):
        world = EnergyWorld(random.Random(seed))
        rewards = []
        for _ in range(episodes):
            state = world.reset()
            episode_reward = 0
            for _ in range(max_steps):
                action = self.choose(state)
                next_state, reward, done = world.step(action)
                self.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(episode_reward)
        self.save()
        return {"episodes": episodes,
                "first_50_mean": round(sum(rewards[:50]) / min(50, len(rewards)), 3),
                "last_50_mean": round(sum(rewards[-50:]) / min(50, len(rewards)), 3),
                "total_steps": self.total_steps}

    def evaluate(self, episodes=100, seed=99):
        world = EnergyWorld(random.Random(seed))
        rewards = []
        for _ in range(episodes):
            state = world.reset()
            score = 0
            for _ in range(50):
                action = self.choose(state, explore=False)
                state, reward, done = world.step(action)
                score += reward
                if done:
                    break
            rewards.append(score)
        return round(sum(rewards) / len(rewards), 3)

    def save(self):
        data = {"q": self.q, "total_steps": self.total_steps,
                "total_reward": self.total_reward, "recent": self.recent}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)

    def load(self):
        data = json.loads(self.path.read_text())
        self.q = data["q"]
        self.total_steps = data["total_steps"]
        self.total_reward = data["total_reward"]
        self.recent = data.get("recent", [])


if __name__ == "__main__":
    agent = AutonomousLearner()
    before = agent.evaluate()
    result = agent.train()
    after = agent.evaluate()
    print(json.dumps({"evaluation_before": before, "training": result,
                      "evaluation_after": after, "improved": after > before}, indent=2))
