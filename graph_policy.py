"""Graph-backed action preferences learned from experienced reward.

Reward symbols are learned observations in the same predictive graph, not
hard-coded action recommendations. The action policy blends graph-predicted
reward with tabular Q values; both must learn from interaction.
"""
import math

class GraphPolicy:
    def __init__(self, graph, blend=0.35):
        self.graph = graph
        self.blend = blend

    @staticmethod
    def decision(state, action):
        return "AT_%d_%d_DO_%s" % (state[0], state[1], action.upper())

    @staticmethod
    def outcome(reward):
        if reward > 0:
            return "REWARD_POSITIVE"
        if reward < -1:
            return "REWARD_TERMINAL"
        return "REWARD_COST"

    def observe(self, state, action, reward):
        self.graph.teach(self.decision(state, action), self.outcome(reward))

    def scores(self, state):
        scores = {}
        for action in ("left", "right", "collect"):
            decision = self.decision(state, action)
            # Avoid creating graph nodes just to query unseen decisions.
            if decision not in self.graph.nodes:
                scores[action] = 0.0
                continue
            weights = self.graph.weights_from(decision)
            # Use only explicit reward edges, not softmax across unrelated nodes.
            positive = weights.get("REWARD_POSITIVE", 0.0)
            cost = weights.get("REWARD_COST", 0.0)
            terminal = weights.get("REWARD_TERMINAL", 0.0)
            scores[action] = 5 * positive - 0.1 * cost - 2 * terminal
        return scores

    def choose(self, state, agent, explore=True):
        if explore and agent.rng.random() < agent.epsilon:
            return agent.rng.choice(agent.ACTIONS)
        graph = self.scores(state)
        q = agent.values(state)
        combined = {a: (1-self.blend)*q[a] + self.blend*graph[a]
                    for a in agent.ACTIONS}
        maximum = max(combined.values())
        best = [a for a in agent.ACTIONS if combined[a] == maximum]
        return agent.rng.choice(best)
