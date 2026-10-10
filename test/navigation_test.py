"""Regression coverage for distance features and detecting repeated two-cell loops."""
import random
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.inspect_dqn_loops import inspect
from src.controller import DQNAgent, DQNConfig
from src.controller.dqn import StateEncoder
from src.environment import BasicPacmanEnvironment
from src.state import ActionSpaceEnum as Action, Position


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.env = BasicPacmanEnvironment(grid_size=6, max_steps=20)
        self.obs = self.env.reset(inner_walls=False)
        self.obs.map.pacman_position = Position(1, 1)
        self.obs.map.pellets = {Position(3, 1)}
        self.obs.map.walls.add(Position(2, 1))
        self.encoder = StateEncoder(6, 20, navigation_features=True)

    def test_distances_follow_corridors_instead_of_crossing_walls(self):
        distances = self.encoder.distance_map(self.obs.map, self.obs.map.pellets)
        self.assertEqual(distances[Position(1, 1)], 4)
        features = self.encoder.navigation(self.obs.map).reshape(4, 3)
        np.testing.assert_array_equal(features[Action.RIGHT], [-1, -1, -1])
        np.testing.assert_allclose(features[Action.DOWN], [0, 0.25, 0])

    def test_food_and_ghost_information_coexists_in_the_same_square(self):
        position = Position(1, 2)
        self.obs.map.pellets = {position}
        self.obs.map.ghost_positions = {position}
        features = self.encoder.navigation(self.obs.map).reshape(4, 3)
        np.testing.assert_array_equal(features[Action.DOWN], [1, 1, 1])

    def test_empty_and_unreachable_targets_have_zero_proximity(self):
        self.obs.map.pellets = {Position(3, 1)}
        self.obs.map.walls.update({Position(2, 1), Position(3, 2), Position(4, 1)})
        features = self.encoder.navigation(self.obs.map).reshape(4, 3)
        np.testing.assert_array_equal(features[Action.DOWN], [0, 0, 0])
        self.obs.map.pellets.clear()
        np.testing.assert_array_equal(self.encoder.navigation(self.obs.map).reshape(4, 3)[Action.DOWN], [0, 0, 0])

    def test_feature_arrays_are_independent_and_have_the_expected_size(self):
        vector = self.encoder(self.obs)
        self.assertEqual(vector.shape, (170,))
        self.assertEqual(vector.dtype, np.float32)
        old = vector.copy()
        self.obs.map.pellets.clear()
        self.assertFalse(np.array_equal(self.encoder(self.obs), vector))
        np.testing.assert_array_equal(vector, old)

    def test_new_checkpoint_reloads_its_feature_configuration(self):
        agent = DQNAgent(self.env, DQNConfig(navigation_features=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            agent.save_model(path)
            loaded = DQNAgent.from_checkpoint(path, self.env)
            self.assertTrue(loaded.config.navigation_features)
            np.testing.assert_array_equal(agent.q_values(self.obs), loaded.q_values(self.obs))

    def test_checkpoints_without_the_new_setting_keep_the_original_encoder(self):
        original = DQNAgent(self.env, DQNConfig())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            original.save_model(path)
            payload = torch.load(path, weights_only=True)
            del payload['config']['navigation_features']
            torch.save(payload, path)
            loaded = DQNAgent.from_checkpoint(path, self.env)
            self.assertFalse(loaded.config.navigation_features)
            self.assertEqual(loaded.encoder.input_size, 158)
            np.testing.assert_array_equal(original.q_values(self.obs), loaded.q_values(self.obs))


class LoopMeasurementTests(unittest.TestCase):
    class Alternator:
        def get_action(self, observation):
            return Action.RIGHT if observation.step_count % 2 == 0 else Action.LEFT

    def test_repeated_two_cell_loop_without_food_is_counted(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=10)
        rng_state = random.getstate()
        result = inspect(self.Alternator(), env, 1, 5)
        self.assertEqual(random.getstate(), rng_state)
        # Step 1 eats a pellet; sustained no-food cycles are counted from step 5.
        self.assertEqual(result['mean_loop_steps'], 6)
        self.assertEqual(result['mean_longest_foodless_run'], 9)
        self.assertEqual(result['episodes_with_loops'], 1)

    def test_short_return_to_previous_square_is_not_a_sustained_loop(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=3)
        result = inspect(self.Alternator(), env, 1, 5)
        self.assertEqual(result['mean_loop_steps'], 0)
        self.assertEqual(result['episodes_with_loops'], 0)


if __name__ == '__main__':
    unittest.main()
