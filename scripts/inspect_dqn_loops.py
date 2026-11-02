"""Measure sustained two-cell oscillations without eating, plus normal game metrics."""
import argparse
import json
from pathlib import Path
import random
import statistics

import torch

from src.controller import DQNAgent
from src.pacman_runner import create_environment


def inspect(agent, env, episodes, seed):
    results = []
    rng_state = random.getstate()
    try:
        for episode in range(episodes):
            random.seed(seed + episode)
            obs = env.reset()
            positions = [obs.map.pacman_position]
            food_counts = [len(obs.map.pellets)]
            loop_steps = loop_run = max_loop_run = foodless = max_foodless = 0
            reward = 0.0
            while not obs.done:
                before_food = len(obs.map.pellets)
                obs = env.step(agent.get_action(obs))
                reward += obs.reward
                foodless = foodless + 1 if len(obs.map.pellets) == before_food else 0
                max_foodless = max(max_foodless, foodless)
                positions.append(obs.map.pacman_position)
                food_counts.append(len(obs.map.pellets))
                # A B A B A: two completed back-and-forth cycles with unchanged food.
                looping = (len(positions) >= 5 and
                           positions[-1] == positions[-3] == positions[-5] and
                           positions[-2] == positions[-4] and positions[-1] != positions[-2] and
                           food_counts[-1] == food_counts[-5])
                loop_steps += looping
                loop_run = loop_run + 1 if looping else 0
                max_loop_run = max(max_loop_run, loop_run)
            results.append({'seed': seed + episode, 'score': obs.score, 'reward': reward,
                            'steps': obs.step_count, 'loop_steps': loop_steps,
                            'longest_loop_continuation': max_loop_run,
                            'longest_foodless_run': max_foodless,
                            'won': not obs.map.pellets and obs.map.pacman_position not in obs.map.ghost_positions})
    finally:
        random.setstate(rng_state)
    return {'episodes': episodes, 'seed': seed,
            'mean_score': statistics.mean(r['score'] for r in results),
            'mean_reward': statistics.mean(r['reward'] for r in results),
            'mean_steps': statistics.mean(r['steps'] for r in results),
            'mean_loop_steps': statistics.mean(r['loop_steps'] for r in results),
            'loop_step_fraction': sum(r['loop_steps'] for r in results) / max(1, sum(r['steps'] for r in results)),
            'episodes_with_loops': sum(r['loop_steps'] > 0 for r in results),
            'mean_longest_foodless_run': statistics.mean(r['longest_foodless_run'] for r in results),
            'win_rate': statistics.mean(r['won'] for r in results), 'episodes_detail': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=40000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    env = create_environment(**payload['environment'])
    agent = DQNAgent.from_checkpoint(args.checkpoint, env)
    result = inspect(agent, env, args.episodes, args.seed)
    result['checkpoint'] = args.checkpoint
    result['checkpoint_episode'] = agent.episodes
    result['navigation_features'] = agent.config.navigation_features
    result['wall_features'] = agent.config.wall_features
    result['local_wall_features'] = agent.config.local_wall_features
    result['double_dqn'] = agent.config.double_dqn
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes_detail'}, indent=2))


if __name__ == '__main__':
    main()
