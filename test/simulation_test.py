"""Deterministic regressions: python -m unittest discover -s test -p '*_test.py' -v"""

import random
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.environment import GhostsPacmanEnvironment
from src.environment.ghosts.strategy import RandomGhostStrategy
from src.state import ActionSpaceEnum as Action, GhostColorEnum, Map, MapFullHash, Position


def make_map(*, ghosts=(), pellets=None, map_type=Map):
    """Create a small board with explicit ghost positions and directions."""
    size = 7
    walls = {
        Position(x, y)
        for x in range(size)
        for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    }
    return map_type(
        size=size,
        walls=walls,
        pellets={Position(4, 3), Position(1, 1), Position(5, 5)} if pellets is None else set(pellets),
        pacman_position=Position(3, 3),
        ghost_positions={position for position, _, _ in ghosts},
        ghost_position_to_color={position: color for position, color, _ in ghosts},
        ghost_position_to_prev_action={position: action for position, _, action in ghosts},
    )


class StateTests(unittest.TestCase):
    def test_numpy_uses_row_y_column_x_for_every_entity(self):
        state = make_map(ghosts=[(Position(4, 2), GhostColorEnum.BLUE, Action.LEFT)])
        flat = state.to_numpy()
        grid = flat.reshape(state.size, state.size)
        self.assertEqual(flat.shape, (49,))
        self.assertEqual(grid[2, 4], -2)
        self.assertEqual(grid[4, 2], 0)
        self.assertEqual(grid[3, 4], 1)
        self.assertEqual(grid[3, 3], 2)
        self.assertEqual(grid[0, 2], -1)

    def test_clone_preserves_type_and_isolates_all_mutable_fields(self):
        position = Position(4, 2)
        original = make_map(
            ghosts=[(position, GhostColorEnum.BLUE, Action.UP)], map_type=MapFullHash,
        )
        snapshot = original.clone()
        clone = original.clone()
        self.assertIsInstance(clone, MapFullHash)
        self.assertEqual(clone, original)
        self.assertEqual(hash(clone), hash(original))
        clone.walls.add(Position(2, 2))
        clone.pellets.clear()
        clone.ghost_positions.clear()
        clone.ghost_position_to_color.clear()
        clone.ghost_position_to_prev_action[position] = Action.DOWN
        clone.pacman_position = Position(2, 3)
        self.assertEqual(original, snapshot)

    def test_full_hash_distinguishes_ghost_directions(self):
        state = make_map(
            ghosts=[(Position(4, 2), GhostColorEnum.BLUE, Action.UP)], map_type=MapFullHash,
        )
        changed = state.clone()
        changed.ghost_position_to_prev_action[Position(4, 2)] = Action.DOWN
        self.assertNotEqual(hash(state), hash(changed))

    def test_full_hash_is_stable_across_python_processes(self):
        script = """
from test.simulation_test import make_map
from src.state import MapFullHash, Position, GhostColorEnum, ActionSpaceEnum
for direction in (None, ActionSpaceEnum.UP, ActionSpaceEnum.DOWN):
    state = make_map(
        ghosts=[(Position(1, 2), GhostColorEnum.BLUE.value, direction)],
        map_type=MapFullHash,
    )
    print(hash((state, ActionSpaceEnum.RIGHT)))
"""
        hashes = []
        for seed in ("1", "2"):
            hashes.append(subprocess.check_output(
                [sys.executable, "-B", "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
            ))
        self.assertEqual(*hashes)


class RewardTests(unittest.TestCase):
    def test_pellet_reward_survives_removal(self):
        env = GhostsPacmanEnvironment.from_map(make_map())
        observation = env.step(Action.RIGHT)
        self.assertEqual(observation.reward, 9)
        self.assertEqual(observation.score, 10)
        self.assertNotIn(Position(4, 3), observation.map.pellets)
        self.assertFalse(observation.done)

    def test_empty_move_costs_one(self):
        env = GhostsPacmanEnvironment.from_map(make_map())
        observation = env.step(Action.UP)
        self.assertEqual(observation.reward, -1)
        self.assertEqual(observation.score, 0)

    def test_wall_costs_fifty_and_does_not_move_pacman(self):
        state = make_map()
        state.walls.add(Position(3, 2))
        env = GhostsPacmanEnvironment.from_map(state)
        observation = env.step(Action.UP)
        self.assertEqual(observation.reward, -50)
        self.assertEqual(observation.map.pacman_position, Position(3, 3))

    def test_collision_before_ghost_movement_is_terminal(self):
        state = make_map(ghosts=[(Position(4, 3), GhostColorEnum.BLUE, Action.RIGHT)])
        env = GhostsPacmanEnvironment.from_map(state, stability_rate=1)
        observation = env.step(Action.RIGHT)
        self.assertEqual(observation.reward, -100)
        self.assertTrue(observation.done)
        self.assertTrue(observation.map.is_terminal())
        self.assertEqual(observation.map.ghost_positions, {Position(4, 3)})

    def test_collision_after_ghost_movement_overrides_food_reward(self):
        state = make_map(ghosts=[(Position(5, 3), GhostColorEnum.BLUE, Action.LEFT)])
        env = GhostsPacmanEnvironment.from_map(state, stability_rate=1)
        observation = env.step(Action.RIGHT)
        self.assertEqual(observation.reward, -100)
        self.assertEqual(observation.score, 10)
        self.assertTrue(observation.done)

    def test_ghost_can_kill_pacman_after_a_blocked_move(self):
        state = make_map(ghosts=[(Position(4, 3), GhostColorEnum.BLUE, Action.LEFT)])
        state.walls.add(Position(3, 2))
        env = GhostsPacmanEnvironment.from_map(state, stability_rate=1)
        observation = env.step(Action.UP)
        self.assertEqual(observation.reward, -100)
        self.assertTrue(observation.done)

    def test_last_pellet_finishes_with_food_reward(self):
        env = GhostsPacmanEnvironment.from_map(make_map(pellets={Position(4, 3)}))
        observation = env.step(Action.RIGHT)
        self.assertEqual(observation.reward, 9)
        self.assertTrue(observation.done)
        self.assertEqual(observation.score, env.max_score)

    def test_step_after_death_has_zero_reward_and_preserves_state(self):
        state = make_map(ghosts=[(Position(4, 3), GhostColorEnum.BLUE, Action.RIGHT)])
        env = GhostsPacmanEnvironment.from_map(state)
        env.step(Action.RIGHT)
        snapshot = env.map.clone()
        observation = env.step(Action.UP)
        self.assertEqual(observation.reward, 0)
        self.assertTrue(observation.done)
        self.assertEqual(observation.step_count, 1)
        self.assertEqual(observation.map, snapshot)


class SimulationTests(unittest.TestCase):
    def test_default_limit_still_has_one_hundred_steps_at_step_one_hundred(self):
        env = GhostsPacmanEnvironment.from_map(make_map(), past_steps=100)
        self.assertFalse(env.done)
        for i in range(100):
            observation = env.step(Action.UP if i % 2 == 0 else Action.DOWN)
            self.assertEqual(observation.done, i == 99)
        self.assertEqual(observation.step_count, 200)

    def test_custom_episode_limit_is_absolute(self):
        env = GhostsPacmanEnvironment.from_map(make_map(), past_steps=73, max_steps=75)
        self.assertFalse(env.step(Action.UP).done)
        self.assertTrue(env.step(Action.DOWN).done)
        self.assertEqual(env.step_count, 75)

    def test_terminal_states_can_be_reconstructed_and_stepped(self):
        death = make_map(ghosts=[(Position(3, 3), GhostColorEnum.BLUE, Action.LEFT)])
        cases = [
            (make_map(pellets=set()), {}),
            (death, {}),
            (make_map(), {"past_steps": 200}),
            (make_map(), {"past_steps": 250}),
            (make_map(), {"done": True}),
        ]
        for state, options in cases:
            with self.subTest(options=options, ghosts=state.ghost_positions):
                env = GhostsPacmanEnvironment.from_map(state, **options)
                snapshot = env.map.clone()
                steps = env.step_count
                observation = env.step(Action.UP)
                self.assertTrue(observation.done)
                self.assertEqual(observation.reward, 0)
                self.assertEqual(observation.step_count, steps)
                self.assertEqual(observation.map, snapshot)

    def test_simulation_preserves_source_map_and_ghost_direction(self):
        state = make_map(ghosts=[(Position(1, 3), GhostColorEnum.BLUE, Action.UP)])
        snapshot = state.clone()
        env = GhostsPacmanEnvironment.from_map(state, stability_rate=1)
        self.assertEqual(next(iter(env.ghosts)).prev_action, Action.UP)
        env.step(Action.RIGHT)
        self.assertEqual(env.map.ghost_positions, {Position(1, 2)})
        self.assertEqual(env.map.ghost_position_to_prev_action, {Position(1, 2): Action.UP})
        self.assertEqual(state, snapshot)

    def test_clone_preserves_episode_settings_score_and_map_type(self):
        env = GhostsPacmanEnvironment.from_map(
            make_map(map_type=MapFullHash), past_steps=73, max_steps=75,
            score=40, stability_rate=0.4,
        )
        clone = env.clone()
        self.assertIsInstance(clone.map, MapFullHash)
        self.assertTrue(clone.full_hash)
        self.assertEqual(clone.map, env.map)
        self.assertEqual(clone.max_steps, 75)
        self.assertEqual(clone.step_count, 73)
        self.assertEqual(clone.score, 40)
        self.assertEqual(clone.max_score, env.max_score)
        self.assertEqual(clone.ghosts_strategy.stability_rate, 0.4)
        clone.step(Action.RIGHT)
        self.assertEqual(env.score, 40)
        self.assertEqual(env.step_count, 73)
        self.assertIn(Position(4, 3), env.map.pellets)
        clone.step(Action.UP)
        self.assertTrue(clone.done)
        self.assertTrue(clone.clone().done)

    def test_reset_and_step_publish_ghost_directions(self):
        rng_state = random.getstate()
        self.addCleanup(random.setstate, rng_state)
        random.seed(4)
        env = GhostsPacmanEnvironment(grid_size=10, num_ghosts=2)
        state = env.reset().map
        self.assertEqual(state.ghost_position_to_prev_action, {g.position: None for g in env.ghosts})
        env.step(Action.UP)
        self.assertEqual(env.map.ghost_position_to_prev_action, {g.position: g.prev_action for g in env.ghosts})

    def test_live_and_cloned_games_match_under_the_same_random_draws(self):
        rng_state = random.getstate()
        self.addCleanup(random.setstate, rng_state)
        random.seed(19)
        # Both ghosts may compete for the same square: move order must survive cloning.
        state = make_map(ghosts=[
            (Position(2, 1), GhostColorEnum.BLUE, Action.RIGHT),
            (Position(4, 1), GhostColorEnum.GREEN, Action.LEFT),
        ])
        env = GhostsPacmanEnvironment.from_map(state, past_steps=10, max_steps=60, score=30)
        clone = env.clone()
        for action in [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP] * 5:
            before_draw = random.getstate()
            actual = env.step(action)
            random.setstate(before_draw)
            simulated = clone.step(action)
            self.assertEqual(actual, simulated)
            if actual.done:
                break

    def test_maps_without_direction_metadata_start_with_no_previous_move(self):
        state = make_map(ghosts=[(Position(1, 3), GhostColorEnum.BLUE, Action.UP)])
        state.ghost_position_to_prev_action.clear()
        env = GhostsPacmanEnvironment.from_map(state)
        self.assertIsNone(next(iter(env.ghosts)).prev_action)

    def test_invalid_step_counts_are_rejected(self):
        for options in ({"past_steps": -1}, {"max_steps": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GhostsPacmanEnvironment.from_map(make_map(), **options)


class GhostStrategyTests(unittest.TestCase):
    def test_all_directions_including_up_honor_stability(self):
        strategy = RandomGhostStrategy(stability_rate=1)
        with patch("src.environment.ghosts.strategy.random.random.choice") as choice:
            for direction in Action:
                self.assertEqual(strategy.get_action(direction, list(Action)), direction)
            choice.assert_not_called()

    def test_blocked_previous_direction_chooses_an_available_action(self):
        strategy = RandomGhostStrategy(stability_rate=1)
        self.assertEqual(strategy.get_action(Action.UP, [Action.RIGHT]), Action.RIGHT)


if __name__ == "__main__":
    unittest.main()
