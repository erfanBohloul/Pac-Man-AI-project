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
