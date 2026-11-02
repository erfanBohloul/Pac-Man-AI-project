import unittest
from unittest.mock import patch

from src.controller import BasicController
from src.controller.mcts import MCTSAgent
from src.environment import BasicPacmanEnvironment
from src.pacman_runner import create_dqn_mcts_controller
from src.state import ActionSpaceEnum


class FixedPolicyController:
    """DQN-shaped controller stub for testing the shared rollout interface."""

    def __init__(self, preferred=ActionSpaceEnum.UP):
        self.preferred = preferred
        self.calls = 0

    def get_action(self, observation):
        self.calls += 1
        legal = observation.map.get_legal_actions()
        return self.preferred if self.preferred in legal else legal[0]


class DQNMCTSIntegrationTests(unittest.TestCase):
    def test_factory_injects_loaded_dqn_into_the_shared_mcts_engine(self):
        environment = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        policy = FixedPolicyController()
        with patch("src.controller.DQNAgent.from_checkpoint", return_value=policy) as load:
            agent = create_dqn_mcts_controller(
                environment, "checkpoint.pt", device="cpu", iterations=4, max_depth=3,
            )

        load.assert_called_once_with("checkpoint.pt", environment, device="cpu")
        self.assertIsInstance(agent, MCTSAgent)
        self.assertIs(agent.rollout_controller, policy)

    def test_dqn_rollout_policy_works_at_multiple_search_budgets(self):
        budgets = ((1, 1), (5, 3), (10, 5), (30, 10))
        for iterations, max_depth in budgets:
            with self.subTest(iterations=iterations, max_depth=max_depth):
                environment = BasicPacmanEnvironment(grid_size=6, max_steps=12)
                observation = environment.reset()
                policy = FixedPolicyController()
                agent = MCTSAgent(
                    environment, policy, iterations=iterations,
                    max_depth=max_depth, seed=5,
                )
                before = agent._state_key(environment)
                with patch.object(agent, "_simulate", wraps=agent._simulate) as simulate:
                    action = agent.get_action(observation)

                self.assertEqual(simulate.call_count, iterations)
                self.assertIn(action, environment.map.get_legal_actions())
                if max_depth > 1:
                    self.assertGreater(policy.calls, 0)
                else:
                    self.assertEqual(policy.calls, 0)
                self.assertEqual(before, agent._state_key(environment))

    def test_vanilla_and_dqn_variants_are_the_same_search_class(self):
        environment = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        vanilla = MCTSAgent(environment, BasicController(legal_only=True))
        hybrid = MCTSAgent(environment, FixedPolicyController())

        self.assertIs(type(vanilla), type(hybrid))
        self.assertNotEqual(type(vanilla.rollout_controller), type(hybrid.rollout_controller))


if __name__ == "__main__":
    unittest.main()
