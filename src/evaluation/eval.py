"""Evaluate policies without learning, using reproducible episode seeds."""
import random
import statistics
from typing import Any

from src.environment import PacmanEnvironment
from src.controller.controller import Controller


def evaluate_algorithm(environment: PacmanEnvironment, controller: Controller,
                       num_episodes=1000, seed=None) -> dict[str, Any]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    scores, steps, rewards, fractions = [], [], [], []
    wins = deaths = 0
    winning_steps = []
    rng_state = random.getstate()
    try:
        for episode in range(num_episodes):
            if seed is not None:
                random.seed(seed + episode)
            observation = environment.reset()
            initial_pellets = len(observation.map.pellets)
            total_reward = 0.0
            while not observation.done:
                action = controller.get_action(observation)
                observation = environment.step(action)
                total_reward += observation.reward
            dead = observation.map.pacman_position in observation.map.ghost_positions
            won = not observation.map.pellets and not dead
            wins += won
            deaths += dead
            if won:
                winning_steps.append(observation.step_count)
            scores.append(observation.score)
            steps.append(observation.step_count)
            rewards.append(total_reward)
            fractions.append(1 - len(observation.map.pellets) / max(1, initial_pellets))
    finally:
        random.setstate(rng_state)
    stats = {
        "Episodes": num_episodes,
        "Mean-score": statistics.mean(scores),
        "Std-score": statistics.stdev(scores) if num_episodes > 1 else 0.0,
        "Mean-reward": statistics.mean(rewards),
        "Mean-steps": statistics.mean(steps),
        "Mean-food-fraction": statistics.mean(fractions),
        "Win-rate": wins / num_episodes,
        "Death-rate": deaths / num_episodes,
    }
    if winning_steps:
        stats["Min-steps-on-win"] = min(winning_steps)
    return stats
