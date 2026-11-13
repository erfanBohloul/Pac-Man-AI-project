# Feature comparison on the final 15 x 15 map

This experiment uses the group's confirmed environment: 15 x 15, four random
persistent ghosts, stability 0.75 and a 500-step limit. Its fixed maze has 140
initial pellets. The runner verifies that all pellets are reachable and records
a SHA-256 of the sorted wall coordinates; grid dimensions alone do not identify
a maze. Old 10 x 10 checkpoints and scores are not reused.

## Conditions and controls

| Variant | Raw wall plane | Four adjacent-wall bits | Input size | Architecture |
| --- | --- | --- | ---: | --- |
| walls | yes | no | 737 | 737 -> 128 -> 128 -> 4 |
| no_walls | no | no | 512 | 512 -> 128 -> 128 -> 4 |
| local_walls | no | yes | 516 | 516 -> 128 -> 128 -> 4 |

Food and Pac-Man planes, six ghost slots, remaining time, stability and all 12
BFS navigation features are present in every condition. BFS and action masks
still use the walls. In particular, blocked directions already have -1 in their
three navigation fields. Adding four explicit bits does not introduce entirely
new obstacle information.

All runs use standard DQN (Double DQN off), 128-unit ReLU hidden layers, Adam
with learning rate 0.0005, gamma 0.98, batch size 64, uniform replay capacity
20000, warmup 1000 steps, updates every four steps, target copies every 250
optimizer updates, SmoothL1, reward scale 0.1 and gradient norm clipping at 10.
Epsilon falls linearly from 1 to 0.05 across 50000 environment steps.

The training budget matches the current notebook: **1000 episodes per model**,
validation every 50 episodes on ten seeds (1000000-1000009). The checkpoint is
selected by validation mean reward. Training seeds are 0, 10000 and 20000, with
disjoint episode-seed ranges. Each of the nine selected models is evaluated on
300 held-out games, seeds 2000000-2000299. The test set never selects checkpoints.
Thus there are 2700 test games but only **three trained models per condition**.

Equal episode budgets need not have equal interaction counts; longer survival
produces more training updates. The input shape changes initialization as well
as trajectories. This experiment compares outcomes of the three representations
under the same training procedure, not one-to-one identical optimization paths.

## Reproduction

From the repository root with project dependencies installed:

```bash
python -m unittest test.final_map_features_test
python -m scripts.compare_final_map_features \
  --train-episodes 1000 --test-episodes 300 --workers 3 \
  --output-dir outputs/dqn/final-map-features
```

Use a fresh output directory. Partial/completed runs are not overwritten.
`experiment.json` records settings and source commit; `report.json` contains all
per-model configurations, validation results, test statistics and timings.
Individual run directories retain checkpoints, episode logs and test details.

The script directly calls the maintained environment and DQN APIs. It does not
use the deleted `cli.py`/`pacman_runner.py`. To make this path importable, public
controller exports no longer import deleted legacy agents, and the environment
package exports its remaining base class and ghost environment. The loop
inspection CLI likewise constructs the ghost environment directly.

Three focused integration tests check input differences, fixed-maze reachability
and fingerprint, RNG preservation, and train/save/load/evaluation agreement.
Legacy test files and scripts which still import the deleted basic environment
or runner require separate migration; this experiment does not claim that the
entire old test suite is passing.

## Interpreting results

Report mean and sample standard deviation across the three trained models.
Episode-score standard deviation within one model is a different statistic.
A counted loop step continues A-B-A-B-A with no food eaten across four moves;
this diagnoses sustained two-cell oscillation, not every possible cycle.

Training timing includes periodic validation and checkpoint writes. Three jobs
run concurrently, so wall times are not isolated throughput benchmarks. Test
metrics are reproduced by the loop inspector; reported test-plus-loop timing
includes both passes. This is a fixed-map experiment, not unseen-map testing.
The notebook's short 1000-episode budget may not converge, and ten validation
games make checkpoint selection noisy. An uncertain result should motivate a
predefined larger-budget experiment, not tuning on the current held-out test.

This is a DQN-only experiment. No MCTS rollout is used, so the shared search/game
RNG issue does not affect this comparison and remains separate integration work.

## Results (2026-11-13)

Entries are mean ± sample standard deviation across three independently trained models. Percentages are in percent units, including their standard deviations.

| Encoding | Score | Win rate (%) | Food collected (%) | Flagged loop steps (%) | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw wall plane | 1004.14 ± 20.19 | 33.33 ± 8.97 | 71.72 ± 1.44 | 7.88 ± 2.68 | 94.20 ± 4.92 |
| No raw wall plane | 1066.11 ± 44.85 | 26.44 ± 9.86 | 76.15 ± 3.20 | 10.59 ± 1.43 | 110.92 ± 7.32 |
| Four local wall bits | 1066.83 ± 39.85 | 17.22 ± 4.53 | 76.20 ± 2.85 | 11.56 ± 1.93 | 112.13 ± 7.15 |

The raw-wall condition had the highest mean win rate (33.33%) and lowest flagged-loop fraction (7.88%) at this budget. The other two conditions collected more food and scored higher, but won less often. Local bits did not improve on simply omitting raw walls: 17.22% versus 26.44% wins, and 11.56% versus 10.59% loop steps.

For the current notebook budget, the raw-wall encoding is the recommended baseline if the priorities are completing games and reducing oscillation. Retain the other conditions as feature-ablation exercises. This is a conditional recommendation from three training seeds, not proof that the raw plane is necessary or universally better. On a single fixed map, the raw wall plane is constant; it can still change optimization through parameterization/initialization.

| Encoding | Seed | Validation-selected episode | Test score | Test win rate | Loop-step fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw wall plane | 0 | 1000 | 1021.93 | 40.67% | 6.76% |
| No raw wall plane | 0 | 800 | 1024.23 | 17.00% | 11.03% |
| Four local wall bits | 0 | 750 | 1022.63 | 20.00% | 13.15% |
| Raw wall plane | 10000 | 950 | 1008.30 | 23.33% | 10.94% |
| No raw wall plane | 10000 | 800 | 1113.43 | 25.67% | 11.75% |
| Four local wall bits | 10000 | 850 | 1100.00 | 19.67% | 12.12% |
| Raw wall plane | 20000 | 1000 | 982.20 | 36.00% | 5.94% |
| No raw wall plane | 20000 | 950 | 1060.67 | 36.67% | 9.00% |
| Four local wall bits | 20000 | 950 | 1077.87 | 12.00% | 9.42% |

The nine training loops took about 91-120 seconds each on this machine, with concurrent jobs and validation included. This is not a portable runtime promise. Timing does not include the two 300-game evaluation passes.

Full settings, per-model results, maze identity, code hashes and timing are in `dqn-final-map-features-validation.json`. Large checkpoints and logs stay in local `outputs/dqn/final-map-features/`. Three targeted integration tests passed. Existing legacy imports/tests outside this experiment still need migration.

These results supersede neither the old 10 x 10 experiments nor their different training budgets: they answer the feature question specifically for the confirmed 15 x 15 environment and current 1000-episode notebook setup. A larger-budget follow-up must use fresh held-out test seeds if the present results inform its design.
