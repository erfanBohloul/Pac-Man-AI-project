import unittest
from types import SimpleNamespace

import numpy as np

from src.controller.mcts import MCTSAgent
from src.environment import BasicPacmanEnvironment
from src.state import ActionSpaceEnum


class FixedPolicy:
    def __init__(self, preferred=ActionSpaceEnum.UP, reward_scale=0.1):
        self.preferred = preferred
        self.config = SimpleNamespace(reward_scale=reward_scale)
        self.calls = 0

    def q_values(self, observation):
        self.calls += 1
        values = np.zeros(4, dtype=np.float32)
        values[int(self.preferred)] = 1.0
        return values


class DQNGuidedMCTSTests(unittest.TestCase):
    def setUp(self):
        self.environment = BasicPacmanEnvironment(grid_size=6, max_steps=20)
        self.observation = self.environment.reset()
        self.policy = FixedPolicy()
        self.agent = MCTSAgent(
            self.environment, iterations=12, max_depth=4, dqn_agent=self.policy,
            seed=7,
        )

    def test_dqn_values_create_action_priors(self):
        actions = self.environment.map.get_legal_actions()
        node = self.agent._make_node(actions, self.observation)

        self.assertGreater(node.priors[ActionSpaceEnum.UP], node.priors[ActionSpaceEnum.DOWN])
        self.assertAlmostEqual(sum(node.priors.values()), 1.0)

    def test_search_returns_legal_action_without_mutating_live_game(self):
        original_state = self.agent._state_key(self.environment)
        action = self.agent.get_action(self.observation)

        self.assertIn(action, self.environment.map.get_legal_actions())
        self.assertEqual(original_state, self.agent._state_key(self.environment))
        self.assertGreater(self.policy.calls, 0)

    def test_depth_limit_bootstraps_in_environment_reward_units(self):
        _, bootstrap = self.agent._rollout(self.environment, self.observation, self.agent.max_depth)

        self.assertAlmostEqual(bootstrap, 10.0)


if __name__ == "__main__":
    unittest.main()
