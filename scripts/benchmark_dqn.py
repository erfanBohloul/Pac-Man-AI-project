"""Compare saved DQN, untrained DQN and both random baselines on held-out seeds.

Run: python -m scripts.benchmark_dqn --checkpoint outputs/dqn/RUN/best.pt
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from src.controller import BasicController, DQNAgent
from src.evaluation import evaluate_algorithm
from src.pacman_runner import create_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--output')
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    env = create_environment(**payload['environment'])
    trained = DQNAgent.from_checkpoint(args.checkpoint, env)
    untrained = DQNAgent(env, replace(trained.config, device='cpu'))
    controllers = {
        'random_all_actions': BasicController(seed=args.seed),
        'random_legal_actions': BasicController(legal_only=True, seed=args.seed),
        'untrained_dqn': untrained,
        'trained_dqn': trained,
    }
    result = {'checkpoint': args.checkpoint, 'environment': payload['environment'],
              'evaluation_seed': args.seed, 'training_seed': trained.config.seed,
              'checkpoint_episode': trained.episodes, 'policies': {}}
    for name, controller in controllers.items():
        result['policies'][name] = evaluate_algorithm(env, controller, args.episodes, seed=args.seed)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / 'benchmark.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
