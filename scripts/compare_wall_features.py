"""Train/evaluate three fixed-maze feature variants on matched training seeds.

Run from the repository root. Existing completed trials are validated and reused;
partial trials require a fresh --output-dir. Logs/checkpoints stay local.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from src.controller import DQNConfig

VARIANTS = {'walls': (True, False), 'no_walls': (False, False), 'local_walls': (False, True)}
ENVIRONMENT = {'environment_name': 'ghosts', 'grid_size': 10, 'max_steps': 200,
               'num_ghosts': 2, 'stability_rate': .75}


def trial(variant, seed, args):
    walls, local = VARIANTS[variant]
    folder = args.output_dir / f'{variant}-seed{seed}'
    if args.reuse_seed0_baselines and seed == 0 and variant != 'local_walls':
        folder = Path('outputs/dqn') / ('navigation-seed0' if walls else 'no-walls-seed0')
    config = DQNConfig(seed=seed, navigation_features=True, wall_features=walls,
                       local_wall_features=local, epsilon_decay_steps=80000)
    expected = {'config': asdict(config), 'environment': ENVIRONMENT,
                'episodes': 6000, 'eval_seed': 1000000, 'eval_episodes': 100}
    if (folder / 'run.json').exists():
        run = json.loads((folder / 'run.json').read_text())
        run['config'] = asdict(DQNConfig(**run['config']))
        if run != expected:
            raise ValueError(f'Existing settings differ: {folder}')
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
        if not rows or rows[-1]['episode'] != 6000 or not (folder / 'best.pt').exists():
            raise ValueError(f'Partial run; use a fresh output directory: {folder}')
        print(f'Reusing completed {variant} seed {seed}', flush=True)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        print(f'Training {variant} seed {seed}', flush=True)
        command = [sys.executable, 'cli.py', 'controller=dqn_local', 'environment=ghosts',
                   f'controller.wall_features={str(walls).lower()}',
                   f'controller.local_wall_features={str(local).lower()}', f'seed={seed}',
                   f'controller.output_dir={folder}', 'num_episodes=1', f'eval_seed={args.eval_seed}']
        with (folder / 'console.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics_path = folder / f'comparison-{args.eval_seed}.json'
    with (folder / 'comparison.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'scripts.inspect_dqn_loops',
                        '--checkpoint', str(folder / 'best.pt'), '--episodes', str(args.eval_episodes),
                        '--seed', str(args.eval_seed), '--output', str(metrics_path)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads(metrics_path.read_text())
    metrics.pop('episodes_detail')
    payload = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
    selected = next(row for row in rows if row['episode'] == payload['episodes'])
    timing = json.loads((folder / 'timing.json').read_text()) if (folder / 'timing.json').exists() else None
    print(f'Finished {variant} seed {seed}: score {metrics["mean_score"]:.1f}, '
          f'wins {metrics["win_rate"]:.1%}, loop steps {metrics["loop_step_fraction"]:.2%}', flush=True)
    return {'variant': variant, 'training_seed': seed, 'settings': expected,
            'input_size': payload['online']['0.weight'].shape[1],
            'parameters': sum(t.numel() for t in payload['online'].values()),
            'final_environment_steps': rows[-1]['environment_steps'],
            'selected_validation': selected['validation'], 'timing': timing, 'evaluation': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10000, 20000])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--eval-seed', type=int, default=70000)
    parser.add_argument('--eval-episodes', type=int, default=300)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/dqn/wall-comparison'))
    parser.add_argument('--reuse-seed0-baselines', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.eval_episodes < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive workers/episodes and distinct seeds are required')
    test = set(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    validation = set(range(1000000, 1000100))
    if test & validation or any(set(range(seed, seed + 6000)) & (test | validation) for seed in args.seeds):
        parser.error('Training, validation and test seed ranges must be disjoint')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, seed) for seed in args.seeds for variant in ['local_walls', 'walls', 'no_walls']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(trial, variant, seed, args) for variant, seed in jobs]
        trials = [future.result() for future in futures]
    aggregate = {}
    for variant in VARIANTS:
        group = [row for row in trials if row['variant'] == variant]
        aggregate[variant] = {}
        for metric in ['mean_score', 'mean_reward', 'win_rate', 'loop_step_fraction', 'mean_steps',
                       'mean_longest_foodless_run']:
            values = [row['evaluation'][metric] for row in group]
            aggregate[variant][metric] = {'mean': statistics.mean(values),
                'sample_std': statistics.stdev(values) if len(values) > 1 else None}
    report = {'experiment': 'Raw wall plane vs no raw walls vs four explicit adjacent-wall bits.',
              'training_seeds': args.seeds, 'eval_seed': args.eval_seed, 'eval_episodes_per_model': args.eval_episodes,
              'workers': args.workers, 'trials': trials, 'aggregate': aggregate,
              'notes': ['Aggregate statistics weight each training seed equally; standard deviation is across trained models.',
                        'Same fixed maze, 6000 episodes, hyperparameters and validation selection rule. Interaction counts vary.',
                        'Navigation features and action masks still use walls in every condition.',
                        'Validation chooses best mean reward; final test is not used to select checkpoints.',
                        'Three training seeds do not establish a universal advantage or unseen-map generalization.',
                        'Concurrent runs share CPU resources. Timings are not isolated speed benchmarks; old runs have null timing.']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
