"""Bridge action experience into Haven's persistent predictive graph.

The graph learns observed state/action/outcome transitions. The action policy
still uses tabular Q-learning; this bridge does not pretend graph predictions
already control decisions.
"""
from haven import Haven
from graph_policy import GraphPolicy
from autonomy import AutonomousLearner, EnergyWorld
import random


class IntegratedHaven:
    def __init__(self, graph_path="haven_state.json", agent_path="agent_state.json",
                 seed=13, epsilon=0.15):
        self.graph = Haven(graph_path)
        self.agent = AutonomousLearner(agent_path, seed=seed, epsilon=epsilon)
        self.policy = GraphPolicy(self.graph)
        self.world = EnergyWorld(random.Random(seed + 1))
        self.state = self.world.reset()
        self.episodes = 0
        self.collections = 0

    @staticmethod
    def symbol(state):
        return "POS_%d_RESOURCE_%d" % state

    @staticmethod
    def action_symbol(state, action):
        return "AT_%d_%d_DO_%s" % (state[0], state[1], action.upper())

    def act(self, explore=True):
        state = self.state
        action = self.policy.choose(state, self.agent, explore=explore)
        next_state, reward, done = self.world.step(action)
        self.agent.learn(state, action, reward, next_state, done)
        # Two graph observations: state -> chosen action, and action -> outcome.
        # These edges are updated from actual experience, not hand-authored rules.
        source = self.symbol(state)
        decision = self.action_symbol(state, action)
        outcome = self.symbol(next_state)
        self.graph.teach(source, decision)
        self.graph.teach(decision, outcome)
        self.policy.observe(state, action, reward)
        if reward > 0:
            self.collections += 1
        event = {"state": state, "action": action, "next_state": next_state,
                 "reward": reward, "done": done,
                 "graph_prediction": self.graph.predict(decision)[0],
                 "graph_policy_scores": self.policy.scores(state)}
        if done:
            self.episodes += 1
            self.state = self.world.reset()
        else:
            self.state = next_state
        self.agent.save()
        return event

    def run(self, steps=100):
        if not 1 <= steps <= 100000:
            raise ValueError("steps must be 1..100000")
        events = [self.act() for _ in range(steps)]
        return {"steps": self.agent.total_steps, "episodes": self.episodes,
                "collections": self.collections, "recent": events[-10:],
                "graph_nodes": len(self.graph.nodes),
                "graph_edges": len(self.graph.edges)}

    def snapshot(self):
        return {"agent_steps": self.agent.total_steps, "episodes": self.episodes,
                "collections": self.collections, "position": self.world.position,
                "resource": self.world.resource, "energy": self.world.energy,
                "graph_nodes": len(self.graph.nodes),
                "graph_edges": len(self.graph.edges),
                "recent_actions": self.agent.recent[-10:]}
