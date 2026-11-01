import random
import unittest
from unittest.mock import patch

from src.controller.basic import BasicController
from src.controller.mcts import MCTSAgent, _Edge, _Node
from src.environment import BasicPacmanEnvironment
from src.state import ActionSpaceEnum


class MCTSControllerTests(unittest.TestCase):
    def test_unvisited_edge_is_selected_without_dividing_by_zero(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        agent = MCTSAgent(env, BasicController(legal_only=True), seed=3)
        node = _Node([ActionSpaceEnum.UP, ActionSpaceEnum.DOWN])
        node.actions[ActionSpaceEnum.UP] = _Edge()
        node.actions[ActionSpaceEnum.DOWN] = _Edge()
        node.actions[ActionSpaceEnum.UP].visits = 4
        node.actions[ActionSpaceEnum.UP].value_sum = 2.0
        node.visits = 4

        action, edge = agent._select_edge(node)

        self.assertEqual(action, ActionSpaceEnum.DOWN)
        self.assertEqual(edge.visits, 0)

    def test_uct_uses_mean_return_and_exploration_bonus(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        agent = MCTSAgent(env, BasicController(legal_only=True), exploration=2.0, seed=1)
        node = _Node([ActionSpaceEnum.UP, ActionSpaceEnum.DOWN])
        node.actions[ActionSpaceEnum.UP] = _Edge()
        node.actions[ActionSpaceEnum.DOWN] = _Edge()
        node.actions[ActionSpaceEnum.UP].visits = 10
        node.actions[ActionSpaceEnum.UP].value_sum = 10.0
        node.actions[ActionSpaceEnum.DOWN].visits = 1
        node.actions[ActionSpaceEnum.DOWN].value_sum = 0.0
        node.visits = 11

        action, _ = agent._select_edge(node)

        # The less-visited edge wins because its UCT exploration bonus is larger.
        self.assertEqual(action, ActionSpaceEnum.DOWN)

    def test_rejects_an_empty_node(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        agent = MCTSAgent(env, BasicController(legal_only=True))
        with self.assertRaisesRegex(ValueError, "no actions"):
            agent._select_edge(_Node([]))

    def test_rollouts_use_the_injected_controller(self):
        class CountingController:
            def __init__(self):
                self.calls = 0

            def get_action(self, observation):
                self.calls += 1
                return random.choice(observation.map.get_legal_actions())

        env = BasicPacmanEnvironment(grid_size=6, max_steps=12)
        controller = CountingController()
        agent = MCTSAgent(env, controller, iterations=4, max_depth=4, seed=2)
        observation = env.reset()
        before = agent._state_key(env)

        action = agent.get_action(observation)

        self.assertIn(action, env.map.get_legal_actions())
        self.assertGreater(controller.calls, 0)
        self.assertEqual(before, agent._state_key(env))

    def test_search_handles_multiple_iteration_and_depth_budgets(self):
        for iterations, max_depth in ((1, 1), (4, 2), (12, 6), (25, 10)):
            with self.subTest(iterations=iterations, max_depth=max_depth):
                env = BasicPacmanEnvironment(grid_size=6, max_steps=12)
                observation = env.reset()
                agent = MCTSAgent(
                    env, BasicController(legal_only=True, seed=8),
                    iterations=iterations, max_depth=max_depth, seed=5,
                )
                with patch.object(agent, "_simulate", wraps=agent._simulate) as simulate:
                    action = agent.get_action(observation)
                self.assertEqual(simulate.call_count, iterations)
                self.assertIn(action, env.map.get_legal_actions())


if __name__ == "__main__":
    unittest.main()
