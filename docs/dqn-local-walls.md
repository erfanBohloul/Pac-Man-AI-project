# DQN with four adjacent-wall features

This preset implements the proposed fixed-maze simplification: omit the full raw
wall plane and append four binary indicators for walls immediately **up, right,
down, left** of Pac-Man (the action enum order). A wall is 1, an open square is 0.
Food and ghosts do not turn an open square into a wall. The other state inputs,
original game rewards, Bellman targets and legal-action masks are unchanged.

On the default 10 x 10 map, the architecture is `266 -> 128 -> 128 -> 4` with ReLU
after both hidden layers: 51,204 trainable parameters. Inputs are 100 food bits,
100 Pac-Man bits, 48 ghost fields, remaining time, ghost stability, 12 navigation
features and the four new wall bits. Ghosts are stochastic with persistence 0.75;
there are two ghosts and a 200-step episode limit. This is standard DQN, with
Double DQN disabled.

The navigation features already mark blocked directions with `[-1, -1, -1]`;
BFS distances and action masks also use the maze. Thus the added bits make
existing local wall information explicit, rather than revealing previously
unobservable obstacles. They are not an anti-loop action rule. The experiment
checks whether this representation helps training. It does not guarantee an
improvement or prove a cause for oscillation.

## Run it

After installing the project dependencies (see [installation](dqn.md#installation)):

The repository includes a small seed-0 inference checkpoint, so teammates can
watch the policy without training first:

```bash
python cli.py controller=dqn_local controller.mode=load \
  controller.model_path=checkpoints/dqn-local/seed0.pt \
  environment=ghosts eval=false framerate=8
```

Its provenance, hyperparameters and SHA-256 are in
`checkpoints/dqn-local/metadata.json`. Seed 0 was chosen in advance for the demo;
the checkpoint within that run was selected by validation mean reward, not by
test scores. To train a fresh model instead:

```bash
python cli.py controller=dqn_local environment=ghosts \
  controller.output_dir=outputs/dqn/local-demo num_episodes=300 eval_seed=70000
```

The preset trains 6000 episodes, validates every 500 episodes on 100 games,
and decays epsilon from 1 to 0.05 over 80000 steps. Other settings match the
navigation baseline: 128 hidden units per layer, learning rate 0.0005, gamma
0.98, batch size 64, replay capacity 20000, learning start at 1000 steps,
an update every four steps, and target copies every 250 optimizer updates.

To view your newly trained model with sound (M toggles mute):

```bash
python cli.py controller=dqn_local controller.mode=load \
  controller.model_path=outputs/dqn/local-demo/best.pt \
  environment=ghosts eval=false framerate=8
```

Loading a checkpoint restores its saved feature configuration. Old checkpoints
without `local_wall_features` still use their original input shape and Q-values.
`controller=dqn` retains its existing default; `controller=dqn_local` selects the
new experimental preset. `DQNAgent` continues to implement `Controller.get_action`
and can serve as a future MCTS rollout controller without retraining during search.

## Reproduce the comparison

```bash
python -m scripts.compare_wall_features --workers 3
```

The runner compares three conditions: raw walls, no raw walls, and no raw walls
plus four adjacent-wall bits. Each is trained with seeds 0, 10000 and 20000 using
the same 6000-episode budget and hyperparameters. Validation seeds 1000000-1000099
select the checkpoint with the highest mean reward. Each selected model is then
evaluated on the same held-out seeds 70000-70299, with no exploration. The report
averages metrics equally over the three independently trained models; standard
deviations describe variation across those models, not uncertainty from 900
independent training runs.

Outputs go to `outputs/dqn/wall-comparison/`. Completed runs are checked against
the expected settings and reused; interrupted training cannot resume exactly,
so use a new `--output-dir` after a partial run. The optional
`--reuse-seed0-baselines` flag uses the previously completed seed-0 navigation
and no-wall runs from their original local output paths; it is not needed in a
fresh checkout. Use `--workers 1` for serial runs on slower machines.

Training durations include periodic validation and checkpoint writes. Concurrent
jobs share CPU capacity, so these timings are not isolated speed benchmarks.
The old seed-0 baseline runs have no instrumented timing. Trials have equal
numbers of episodes; environment-step counts and selected checkpoint episodes
can differ. Three training seeds on one fixed maze still do not establish
universal performance or generalization to new layouts.

## Measured results (2026-10-12)

Three training seeds per condition, 300 test episodes per trained model (900 evaluations per condition, sharing test episode seeds). Values below are mean ± sample standard deviation across the three trained models.

| Metric | Raw walls (362 inputs) | No raw walls (262 inputs) | Local wall bits (266 inputs) |
| --- | ---: | ---: | ---: |
| Mean score | 486.29 ± 14.29 | 511.64 ± 35.78 | 512.57 ± 20.35 |
| Win rate (%) | 55.11 ± 4.14 | 59.89 ± 5.42 | 60.89 ± 5.48 |
| Flagged loop steps / all steps (%) | 1.03 ± 0.32 | 3.40 ± 2.44 | 3.44 ± 2.33 |
| Mean episode length | 66.13 ± 1.71 | 91.76 ± 26.79 | 90.27 ± 20.47 |

The local-bit model averaged 60.89% wins, versus 59.89% without raw walls and 55.11% with raw walls. Its loop-step fraction (3.44%) was similar to no raw walls (3.40%), and above raw walls (1.03%). These three runs do not demonstrate that the four bits fix oscillation or establish a reliable win-rate improvement over omitting raw walls alone. Keep the existing default and offer the compact model as an experimental preset.

Per-model results (validation-selected checkpoints, not selected by test performance):

| Variant | Training seed | Selected episode | Mean score | Win rate | Loop-step fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_walls | 0 | 1000 | 522.07 | 65.00% | 4.42% |
| walls | 0 | 4500 | 499.37 | 57.67% | 0.74% |
| no_walls | 0 | 1000 | 539.63 | 61.00% | 5.78% |
| local_walls | 10000 | 4500 | 489.20 | 54.67% | 0.78% |
| walls | 10000 | 5500 | 488.47 | 57.33% | 0.97% |
| no_walls | 10000 | 6000 | 471.33 | 54.00% | 0.90% |
| local_walls | 20000 | 1000 | 526.43 | 63.00% | 5.12% |
| walls | 20000 | 4000 | 471.03 | 50.33% | 1.37% |
| no_walls | 20000 | 1000 | 523.97 | 64.67% | 3.51% |

The three local-bit training loops took 285.41, 281.97 and 276.99 seconds, respectively, including periodic validation. Other trials ran concurrently, so timings should not be used as a controlled speed comparison. The committed sample is the seed-0 checkpoint selected at episode 1000, from a completed 6000-episode run.

All 53 tests passed. A headless Pygame playback of the bundled sample completed with sound enabled (92 steps, score 560); this checks renderer/audio integration, not speaker quality. The full configurations, per-seed metrics, validation results and timings are in `dqn-local-walls-validation.json`. Training logs and the other weights remain local.
