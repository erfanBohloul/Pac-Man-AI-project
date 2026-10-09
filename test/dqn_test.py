import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.controller.dqn import DQNAgent, DQNConfig, ReplayBuffer, StateEncoder, bellman_targets
from src.environment import BasicPacmanEnvironment, GhostsPacmanEnvironment
from src.evaluation import evaluate_algorithm
from src.state import ActionSpaceEnum as Action, GhostColorEnum, Position


class EncoderTests(unittest.TestCase):
    def test_preserves_hidden_food_direction_identity_and_time(self):
        env = GhostsPacmanEnvironment(grid_size=6, num_ghosts=0, max_steps=100)
        obs = env.reset()
        position = Position(1, 2)
        obs.map.ghost_positions = {position}
        obs.map.ghost_position_to_color = {position: GhostColorEnum.BLUE}
        obs.map.ghost_position_to_prev_action = {position: Action.UP}
        obs.map.pellets.add(position)
        encoder = StateEncoder(6, 100, 0.75)
        encoded = encoder(obs)
        self.assertEqual(encoded.shape, (158,))
        self.assertEqual(encoded.dtype, np.float32)
        board = encoded[:108].reshape(3, 6, 6)
        self.assertEqual(board[1, 2, 1], 1)
        blue = encoded[108:116]
        np.testing.assert_allclose(blue, [1, 0.2, 0.4, 1, 0, 0, 0, 0])
        self.assertEqual(encoded[-1], 0.75)
        self.assertEqual(encoded[-2], 1)
        obs.step_count = 100
        self.assertEqual(encoder(obs)[-2], 0)
        obs.map.ghost_position_to_prev_action[position] = Action.DOWN
        self.assertNotEqual(encoder(obs)[111], encoded[111])
        self.assertEqual(encoded[-2], 1)  # Old arrays remain independent.

    def test_rejects_a_different_grid(self):
        with self.assertRaisesRegex(ValueError, 'grid sizes'):
            StateEncoder(8, 200)(BasicPacmanEnvironment(grid_size=6).reset())

    def test_replay_owns_snapshots_and_evicts_oldest(self):
        replay = ReplayBuffer(2)
        state = np.array([1, 2], dtype=np.float32)
        mask = np.ones(4, dtype=bool)
        replay.add(state, 1, 9, state, False, mask)
        state[:] = 0
        mask[:] = False
        self.assertEqual(replay.items[0][0][0], 1)
        self.assertTrue(replay.items[0][-1].all())
        replay.add(state, 2, -1, state, False, mask)
        replay.add(state, 3, -100, state, True, mask)
        self.assertEqual(len(replay), 2)
        batch = replay.sample(2, random.Random(0))
        self.assertEqual(set(batch[1]), {2, 3})


class LearningTests(unittest.TestCase):
    def test_bellman_targets_mask_walls_terminal_and_trapped_states(self):
        rewards = torch.tensor([1., -10., 2.])
        values = torch.tensor([[100., 3., 5., 4.]] * 3)
        masks = torch.tensor([[False, True, True, True], [True] * 4, [False] * 4])
        dones = torch.tensor([False, True, False])
        targets = bellman_targets(rewards, dones, values, masks, 0.5)
        torch.testing.assert_close(targets, torch.tensor([3.5, -10., 2.]))
        online = torch.tensor([[999., 10., 0., 1.]] * 3)
        double_targets = bellman_targets(rewards, dones, values, masks, 0.5, online)
        torch.testing.assert_close(double_targets, torch.tensor([2.5, -10., 2.]))

    def test_optimizer_learns_and_target_syncs_only_at_interval(self):
        env = BasicPacmanEnvironment(grid_size=6, max_steps=20)
        agent = DQNAgent(env, DQNConfig(batch_size=4, target_update_every=2))
        state = agent.encoder(env.reset())
        for _ in range(4):
            agent.replay.add(state, Action.UP, 9, state, True, np.ones(4, dtype=bool))
        before = [p.clone() for p in agent.online.parameters()]
        self.assertTrue(np.isfinite(agent.optimize()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.target.parameters())))
        agent.optimize()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters())))
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))

    def test_exploration_and_greedy_actions_respect_walls(self):
        env = BasicPacmanEnvironment(grid_size=6)
        obs = env.reset()
        obs.map.pacman_position = Position(1, 1)
        agent = DQNAgent(env)
        with patch.object(agent, 'q_values', return_value=np.array([100., 2., 1., 200.])):
            self.assertEqual(agent.get_action(obs), Action.RIGHT)
            for _ in range(20):
                self.assertIn(agent.select_action(obs, epsilon=1), [Action.RIGHT, Action.DOWN])
        self.assertEqual(agent.epsilon, 1)
        agent.environment_steps = agent.config.epsilon_decay_steps
        self.assertAlmostEqual(agent.epsilon, 0.05)

    def test_train_save_reload_and_evaluation_do_not_update_policy(self):
        env = GhostsPacmanEnvironment(grid_size=6, num_ghosts=1, max_steps=10)
        agent = DQNAgent(env, DQNConfig(batch_size=2, learning_starts=2,
                                      train_every=1, target_update_every=2))
        with tempfile.TemporaryDirectory() as directory:
            best = agent.train(env, 3, directory, eval_every=2, eval_episodes=2)
            self.assertTrue(best.exists())
            self.assertTrue((Path(directory) / 'last.pt').exists())
            rows = [json.loads(line) for line in (Path(directory) / 'training.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertGreater(agent.gradient_steps, 0)
            loaded = DQNAgent.from_checkpoint(Path(directory) / 'last.pt', env)
            obs = env.reset()
            np.testing.assert_array_equal(agent.q_values(obs), loaded.q_values(obs))
            self.assertEqual(agent.get_action(obs), loaded.get_action(obs))
            weights = [p.clone() for p in loaded.online.parameters()]
            rng = random.getstate()
            metrics = evaluate_algorithm(env, loaded, 3, seed=54321)
            self.assertEqual(rng, random.getstate())
            self.assertEqual(metrics, evaluate_algorithm(env, loaded, 3, seed=54321))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(weights, loaded.online.parameters())))
            self.assertEqual(len(loaded.replay), 0)
            with self.assertRaisesRegex(ValueError, 'environment mismatch'):
                DQNAgent.from_checkpoint(best, BasicPacmanEnvironment(grid_size=6, max_steps=10))
            with self.assertRaises(FileExistsError):
                agent.train(env, 1, directory)

    def test_spawn_is_free_for_different_map_sizes(self):
        for environment in (BasicPacmanEnvironment, GhostsPacmanEnvironment):
            for size in (6, 8, 10, 12):
                with self.subTest(environment=environment, size=size):
                    state = environment(grid_size=size).reset().map
                    self.assertNotIn(state.pacman_position, state.walls)
                    self.assertNotIn(state.pacman_position, state.pellets)

    def test_training_rejects_overlapping_validation_seeds(self):
        env = BasicPacmanEnvironment(grid_size=6)
        agent = DQNAgent(env, DQNConfig(seed=10))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'seed ranges'):
                agent.train(env, 20, directory, eval_seed=15)


class VisualTests(unittest.TestCase):
    def test_real_drawer_loads_sprites_outside_repository(self):
        import pygame
        from src.drawer import PygameDrawer
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
        }):
            try:
                os.chdir(directory)
                env = GhostsPacmanEnvironment(grid_size=6, num_ghosts=2)
                obs = env.reset()
                drawer = PygameDrawer(grid_size=6, cell_size=30, framerate=0)
                drawer.draw(obs.map, Action.UP)
                self.assertEqual(drawer.screen.get_size(), (180, 180))
                pixels = pygame.surfarray.array3d(drawer.screen)
                self.assertGreater(len(np.unique(pixels.reshape(-1, 3), axis=0)), 10)
            finally:
                pygame.quit()
                os.chdir(cwd)

    def test_visual_runner_uses_reset_observation_for_next_episode(self):
        from src.pacman_runner import run_algorithm
        env = BasicPacmanEnvironment(grid_size=6, max_steps=1)
        controller = Mock()
        seen = []
        def get_action(obs):
            seen.append((obs.step_count, obs.done, len(obs.map.pellets)))
            return Action.UP
        controller.get_action.side_effect = get_action
        with patch('pygame.event.get', return_value=[]), patch('pygame.display.set_caption'):
            run_algorithm(env, Mock(), controller, display_episodes=2)
        self.assertEqual(seen, [(0, False, 14), (0, False, 14)])

    def test_closing_window_does_not_take_another_action(self):
        import pygame
        from src.pacman_runner import run_algorithm
        controller, drawer = Mock(), Mock()
        with patch('pygame.event.get', return_value=[pygame.event.Event(pygame.QUIT)]):
            run_algorithm(BasicPacmanEnvironment(), drawer, controller)
        controller.get_action.assert_not_called()
        drawer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
