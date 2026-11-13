"""Checks for the final-map DQN experiment, independent of removed legacy APIs."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from scripts.compare_final_map_features import ENV, VARIANTS, map_identity
from scripts.inspect_dqn_loops import inspect
from src.controller import DQNAgent, DQNConfig
from src.environment import GhostsPacmanEnvironment
from src.evaluation import evaluate_algorithm


class FinalMapTests(unittest.TestCase):
    def test_encodings_differ_only_by_the_declared_wall_inputs(self):
        env = GhostsPacmanEnvironment(**ENV)
        obs = env.reset()
        vectors = {}
        for variant, (walls, local) in VARIANTS.items():
            agent = DQNAgent(env, DQNConfig(navigation_features=True, wall_features=walls,
                                           local_wall_features=local))
            vectors[variant] = agent.encoder(obs)
        self.assertEqual([len(vectors[v]) for v in VARIANTS], [737, 512, 516])
        np.testing.assert_array_equal(vectors['walls'][225:], vectors['no_walls'])
        np.testing.assert_array_equal(vectors['local_walls'][:-4], vectors['no_walls'])
        expected = [obs.map._state_after_action(a) in obs.map.walls for a in range(4)]
        np.testing.assert_array_equal(vectors['local_walls'][-4:], expected)

    def test_map_identity_is_stable_and_preserves_random_stream(self):
        random.seed(144)
        before = random.getstate()
        first = map_identity(GhostsPacmanEnvironment(**ENV))
        self.assertEqual(random.getstate(), before)
        second = map_identity(GhostsPacmanEnvironment(**ENV))
        self.assertEqual(first, second)
        self.assertTrue(first['all_food_reachable'])
        self.assertEqual(first['initial_food'], 140)

    def test_small_training_roundtrip_and_two_evaluators_agree(self):
        # Short horizon keeps this an integration check, not another training experiment.
        env = GhostsPacmanEnvironment(grid_size=15, num_ghosts=4, max_steps=8)
        agent = DQNAgent(env, DQNConfig(navigation_features=True, wall_features=False,
            local_wall_features=True, learning_starts=1, batch_size=2, train_every=1))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = agent.train(env, 3, directory, eval_every=2, eval_episodes=2)
            policy = DQNAgent.from_checkpoint(checkpoint, env)
            self.assertEqual(policy.encoder.input_size, 516)
            rng = random.getstate()
            metrics = evaluate_algorithm(env, policy, 3, seed=2000000)
            loops = inspect(policy, env, 3, 2000000)
            self.assertEqual(metrics['Mean-score'], loops['mean_score'])
            self.assertEqual(metrics['Win-rate'], loops['win_rate'])
            self.assertEqual(random.getstate(), rng)
            timing = json.loads((Path(directory)/'timing.json').read_text())
            self.assertGreater(timing['total_seconds'], 0)
            self.assertLessEqual(timing['validation_seconds'], timing['total_seconds'])
            self.assertGreater(agent.gradient_steps, 0)


if __name__ == '__main__':
    unittest.main()
