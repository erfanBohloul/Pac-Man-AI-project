"""Compare three DQN encodings on the final maze without the removed CLI.

Run from the repository root: python -m scripts.compare_final_map_features
Defaults match the notebook's 1000-episode training budget and epsilon schedule.
Each subprocess owns a model, environment and random streams.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
from time import perf_counter

import torch

from scripts.inspect_dqn_loops import inspect
from src.controller import DQNAgent, DQNConfig
from src.environment import GhostsPacmanEnvironment
from src.evaluation import evaluate_algorithm

VARIANTS = {'walls': (True, False), 'no_walls': (False, False), 'local_walls': (False, True)}
ENV = dict(grid_size=15, num_ghosts=4, max_steps=500, stability_rate=.75, full_hash=True)


def map_identity(env):
    rng = random.getstate()
    try:
        random.seed(0)
        state = env.reset().map
        walls = sorted((p.x, p.y) for p in state.walls)
        from src.controller.dqn import StateEncoder
        reachable = StateEncoder.distance_map(state, [state.pacman_position])
        if any(p not in reachable for p in state.pellets):
            raise ValueError('Maze contains food unreachable from the initial Pac-Man position')
        return {'wall_sha256': hashlib.sha256(json.dumps(walls).encode()).hexdigest(),
                'walls': walls, 'initial_food': len(state.pellets), 'all_food_reachable': True}
    finally:
        random.setstate(rng)


def trial(variant, seed, settings):
    torch.set_num_threads(1)
    walls, local = VARIANTS[variant]
    env = GhostsPacmanEnvironment(**ENV)
    identity = map_identity(env)
    config = DQNConfig(seed=seed, navigation_features=True, wall_features=walls,
                       local_wall_features=local, epsilon_decay_steps=50000)
    folder = Path(settings['output_dir']) / f'{variant}-seed{seed}'
    if folder.exists():
        raise FileExistsError(f'Use a fresh output directory: {folder}')
    folder.mkdir(parents=True)
    print(f'Start {variant} seed {seed}', flush=True)
    started = perf_counter()
    agent = DQNAgent(env, config)
    best = agent.train(env, settings['train_episodes'], folder, eval_every=50,
                       eval_episodes=10, eval_seed=1000000)
    training_seconds = perf_counter() - started
    policy = DQNAgent.from_checkpoint(best, env)
    evaluation_started = perf_counter()
    metrics = evaluate_algorithm(env, policy, settings['test_episodes'], seed=2000000)
    loops = inspect(policy, env, settings['test_episodes'], 2000000)
    evaluation_seconds = perf_counter() - evaluation_started
    if metrics['Mean-score'] != loops['mean_score'] or metrics['Win-rate'] != loops['win_rate']:
        raise AssertionError('Evaluation/loop inspection must reproduce the same greedy policy results')
    details = loops.pop('episodes_detail')
    (folder/'test-episodes.json').write_text(json.dumps(details, indent=2)+'\n')
    rows = [json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
    selected = next(row for row in rows if row['episode'] == policy.episodes)
    report = {'variant': variant, 'training_seed': seed, 'environment': ENV,
        'map_identity': identity, 'config': asdict(config), 'input_size': policy.encoder.input_size,
        'parameters': sum(p.numel() for p in policy.online.parameters()),
        'checkpoint': str(best), 'selected_episode': policy.episodes,
        'selected_validation': selected['validation'], 'final_environment_steps': rows[-1]['environment_steps'],
        'timing': json.loads((folder/'timing.json').read_text()),
        'training_with_setup_seconds': training_seconds,
        'test_plus_loop_inspection_seconds': evaluation_seconds,
        'metrics': metrics, 'loops': loops}
    (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Finished {variant} seed {seed}: score={metrics["Mean-score"]:.1f}, '
          f'wins={metrics["Win-rate"]:.1%}, loops={loops["loop_step_fraction"]:.2%}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-episodes', type=int, default=1000)
    parser.add_argument('--test-episodes', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,10000,20000])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/dqn/final-map-features'))
    args = parser.parse_args()
    if min(args.train_episodes, args.test_episodes, args.workers) <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive counts and distinct training seeds are required')
    ranges = [(s,s+args.train_episodes) for s in args.seeds] + [(1000000,1000010),(2000000,2000000+args.test_episodes)]
    if any(max(a,c)<min(b,d) for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:]):
        parser.error('Training/validation/test seed ranges must be disjoint')
    if args.output_dir.exists():
        parser.error('Use a fresh output directory; completed results are never overwritten')
    args.output_dir.mkdir(parents=True)
    settings=vars(args).copy(); settings['output_dir']=str(args.output_dir)
    provenance={'settings':settings, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'torch_version':torch.__version__, 'scope':'DQN-only feature comparison; no MCTS simulation used.'}
    (args.output_dir/'experiment.json').write_text(json.dumps(provenance,indent=2)+'\n')
    jobs=[(v,s) for s in args.seeds for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(trial,v,s,settings) for v,s in jobs]
        results=[future.result() for future in futures]
    aggregates={}
    for variant in VARIANTS:
        group=[r for r in results if r['variant']==variant]
        aggregates[variant]={}
        for label, values in {
            'score':[r['metrics']['Mean-score'] for r in group],
            'reward':[r['metrics']['Mean-reward'] for r in group],
            'win_rate':[r['metrics']['Win-rate'] for r in group],
            'food_fraction':[r['metrics']['Mean-food-fraction'] for r in group],
            'loop_fraction':[r['loops']['loop_step_fraction'] for r in group],
            'training_seconds':[r['timing']['total_seconds'] for r in group],
        }.items():
            aggregates[variant][label]={'mean':statistics.mean(values),
                'sample_std':statistics.stdev(values) if len(values)>1 else None}
    report={**provenance,'results':results,'aggregate':aggregates,
        'limitations':['Same fixed maze; no unseen-maze generalization measured.',
            '1000-episode default is the current notebook budget, not a guarantee of convergence.',
            'Equal episode counts, validation rule and hyperparameters; interaction counts/initial weights differ.',
            'Three trained models per condition; shared test seeds do not create 900 independent training runs.',
            'Concurrent processes share CPU capacity; timings are not isolated speed comparisons.',
            'BFS distances and legal-action masking use walls in every variant.']}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregates,indent=2),flush=True)


if __name__ == '__main__':
    main()
