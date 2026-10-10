# DQN for Pac-Man

This implementation learns four action values from the structured game state.
It supports both environments, standalone training, greedy evaluation and Pygame
playback. Training is independent of the renderer and does not open a window.

## Installation

Use Python 3.12 or newer with compatible PyTorch/Pygame wheels. The local run was
verified with Python 3.12.14 on macOS. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
```

`pyproject.toml` includes NumPy, PyTorch, Pygame, Hydra, tqdm, TensorBoard (used by
the legacy REINFORCE controller), and Pillow (GIF export). No Gym, Atari ROMs or
external training dataset is needed. Exact package versions used in the local
verification are recorded in `requirements-macos-py312.txt`. To reproduce that
environment, install it with `pip install -r requirements-macos-py312.txt` before
`pip install -e .`; it is a macOS/Python 3.12 snapshot, not a universal lockfile.

## Train and evaluate

A small deterministic map is useful for checking that training works:

```bash
python cli.py controller=dqn environment=basic grid_size=6 max_steps=80 \
  controller.train_episodes=600 controller.epsilon_decay_steps=20000 \
  controller.eval_every=50 controller.eval_episodes=3 \
  controller.output_dir=outputs/dqn/basic-demo num_episodes=200
```

For the full default map with two ghosts:

```bash
python cli.py controller=dqn environment=ghosts \
  controller.train_episodes=6000 controller.epsilon_decay_steps=80000 \
  controller.eval_every=500 controller.eval_episodes=100 \
  controller.output_dir=outputs/dqn/ghosts-demo num_episodes=300
```

Use a new output directory for every run. Without an override, Hydra generates a
timestamped path. Other options live in `config/controller/dqn.yaml`. `seed`
controls initialization, exploration and the training episodes. Validation uses
seeds beginning at `controller.validation_seed` (1000000); final evaluation uses
`eval_seed` (20000). Keep evaluation seeds outside the training seed range when
increasing the number of training episodes. The trainer rejects overlapping training and validation seed ranges.

The output directory contains:

- `run.json`: environment, hyperparameters, training and validation settings.
- `training.jsonl`: episode score, actual unscaled reward, epsilon and mean loss;
  validation metrics are included at each evaluation interval.
- `best.pt`: the policy with the highest mean validation reward.
- `last.pt`: the policy at the last evaluation interval, including the final episode.

After training, the CLI loads `best.pt` for evaluation or display. A checkpoint
contains the network architecture, encoder version, environment specification,
Q-network weights, target weights and training counters. These are inference
checkpoints: optimizer, replay and RNG states are not saved, so exact training
resumption is not implemented. A missing load path or mismatched environment
raises an error instead of silently training a fresh model.

Load and evaluate without training:

```bash
python cli.py controller=dqn controller.mode=load \
  controller.model_path=outputs/dqn/ghosts-demo/best.pt \
  environment=ghosts num_episodes=300 eval_seed=30000
```

Map size, maximum episode steps, ghost count and stability must match the saved
model. CPU is the default for this small network. `controller.device=mps`,
`cuda`, or `auto` is also available if the installed PyTorch supports the device;
the verification report covers CPU only.

## Watch the trained policy

```bash
python cli.py controller=dqn controller.mode=load \
  controller.model_path=outputs/dqn/ghosts-demo/best.pt \
  environment=ghosts eval=false framerate=8
```

Close the window to stop, or add `display_episodes=3` for a bounded demonstration.
The title bar displays the policy name, score and episode step. For the small
basic model, also pass `environment=basic grid_size=6 max_steps=80` and its model
path. The runner resets the observation between episodes, so a new game never
starts from the previous terminal observation.

Export an episode using the actual Pygame renderer:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m scripts.record_dqn \
  --checkpoint outputs/dqn/ghosts-demo/best.pt \
  --output outputs/dqn/ghosts-demo/gameplay.gif --seed 30000
```

Omit the SDL variables to render with a desktop window. Recording reports the
score and episode length. A single GIF is illustrative, not a performance test.

## Algorithm and state representation

The network is an MLP with two 128-unit ReLU hidden layers and four unbounded Q
outputs (Up, Right, Down, Left). Input consists of three separate flattened maps
for walls, pellets and Pac-Man, six fixed ghost slots, remaining-time fraction
and ghost stability. Each ghost slot contains presence, normalized x/y and a
five-way direction encoding (including no previous direction). Food under a
ghost remains visible in its own input layer. No future simulator outcomes or
hand-coded action policy are supplied to the network.

New CLI training runs also enable `navigation_features=true`. Twelve additional
inputs describe each candidate move: whether that square contains food,
`1 / (1 + distance_to_food)`, and `1 / (1 + distance_to_ghost)`. Distances are
computed by multi-source BFS on the currently observed maze, respecting walls;
unreachable/missing targets have proximity zero, and wall moves use -1. These
are engineered state features, so the resulting agent is **DQN with navigation
features**. The Q-network still chooses the move; there is no scripted food
policy, anti-reversal rule, inference-time random escape, or MCTS search.

The original flattened-map MLP sometimes assigned higher Q-values to repeated
up/down or left/right actions after nearby food was eaten. Supplying spatial
distance features makes progress toward the remaining food easier to learn.
The environment rewards and Bellman update are unchanged. Set
`controller.navigation_features=false` for an ablation of these extra inputs.
Saved checkpoints retain their encoder setting; older checkpoints lacking this
field still load with the original input shape and behavior. Loading an old
model does **not** upgrade its policy: use the newly trained checkpoint below.

The default algorithm is DQN: uniform experience replay, an independently held
target network, epsilon-greedy exploration with linear decay, Adam, Huber loss
and gradient clipping. The target network is copied every 250 optimizer updates.
Updates run every four environment steps after replay warm-up. Rewards are
scaled by 0.1 for learning; metrics and scores retain their original units.
`double_dqn=true` enables the optional Double DQN action-selection target.

Wall actions are masked in both action selection and target maximization by
default. `mask_actions=false` restores the original four-action setting. Empty
legal-action sets bootstrap to zero. Terminal transitions also bootstrap to
zero, including the configured finite episode deadline; remaining time is
observed, so the deadline is part of this task rather than an external cutoff.

Replay entries own their state arrays. This is necessary because the game's
`Observation.map` refers to mutable environment state. `get_action()` performs
greedy inference without updates or exploration. `q_values()` exposes the four
values for a future MCTS integration, in **scaled** reward units; a search using
raw game rewards must divide these values by `reward_scale`.

The implementation follows the standard ingredients explained in the
[official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
adapted to this repository's environment and finite-horizon observations.

## Verification and fair comparisons

```bash
python -m unittest discover -s test -p '*_test.py' -v
python -m scripts.benchmark_dqn \
  --checkpoint outputs/dqn/ghosts-demo/best.pt --episodes 300 --seed 30000
```

The benchmark compares the trained policy with the same architecture at its
initial weights, the original random policy, and a random policy using only
legal actions. The latter shares DQN's default action mask. Every policy is
assessed using the same episode-seed range. Results include score standard
deviation, reward, steps, food fraction, win rate and death rate, saved beside
the checkpoint as `benchmark.json`.

Tests cover encoding, replay independence/eviction, masked Bellman and Double
DQN targets, terminal handling, actual parameter updates, target synchronization,
epsilon/greedy behavior, checkpoint equality and compatibility checks, evaluation
without learning, free spawn cells, real headless rendering, GUI reset and close
handling. Existing simulator regressions are also included.

See `dqn-validation.json` for the measured local runs. Training used one model
seed per configuration. The basic map has a fixed initial state, so repeated
evaluations demonstrate reliable execution of that one learned route, not
unseen-map generalization. The ghost map varies spawn positions and ghost moves;
the original model improved score but did not reliably clear the map. The
navigation-feature update is reported separately below. Multi-seed training and
unseen-map experiments remain future work.

Weights, logs and GIFs stay in the ignored local `outputs/` directory. Code,
configuration, tests, this guide and the compact validation report are committed.

## Original flattened-input results (2026-10-09)

| Environment | Episodes evaluated | Random (all actions) | Random (legal actions) | Untrained DQN | Trained DQN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic, 6 x 6, 80-step limit | 200 | 119.70 | 132.75 | 10.00 | 140.00 |
| Two ghosts, 10 x 10, 200-step limit | 300 | 108.80 | 111.83 | 64.73 | 201.13 |

Values are mean game scores. The basic policy collected all 14 pellets in 14
steps (100% completion on its fixed map). The ghost policy collected 33.5% of
food on average, with no complete wins in these 300 episodes. Its mean score was
about 80% above the random-legal baseline; that is evidence of learning, not a
claim of solving Pac-Man or generalizing to unseen layouts.

The basic run trained for 600 episodes and selected episode 100. The ghost run
trained for 12000 episodes and selected episode 5000, using validation seeds
1000000-1000099 and test seeds 30000-30299. DQN used standard targets, not the
optional Double DQN variant. All 35 unit/regression tests passed. The actual
macOS Pygame window was also exercised with both basic and ghost policies.

The original local demonstration can be launched immediately after activation:

```bash
python cli.py controller=dqn controller.mode=load \
  controller.model_path=outputs/dqn/ghosts-10-verified-seed0/best.pt \
  environment=ghosts eval=false framerate=8
```

Local GIFs are `outputs/dqn/basic-6-seed0/gameplay.gif` and
`outputs/dqn/ghosts-10-verified-seed0/gameplay.gif`. The ghost GIF illustrates
seed 0 (score 400 over 110 moves) and is separate from the evaluation set.

## Diagnosing oscillation (2026-10-10)

The reported up/down behavior was reproduced with the original checkpoint and
seed 0: with 20 pellets remaining, its greedy policy repeatedly alternated
between `(4, 2)` and `(4, 3)`, eventually dying at step 110. This was a learned
Q-value problem, not an animation reset or action-coordinate bug. Small positive
Q-value differences favored repeating the same moves despite the -1 step cost.

The new training preset adds the distance features described above. It keeps
standard DQN targets (`double_dqn=false`), the same game rewards and the ability
to reverse direction when necessary. Both feature construction and old-model
loading are covered by regression tests. The new checkpoint must be used to see
the change; loading yesterday's weights preserves yesterday's policy.

Measure behavior with:

```bash
python -m scripts.inspect_dqn_loops \
  --checkpoint outputs/dqn/navigation-seed0/best.pt \
  --episodes 300 --seed 40000 \
  --output outputs/dqn/navigation-seed0/loops.json
```

A loop step is the continuation of an `A, B, A, B, A` two-cell alternation with
no pellet eaten across those four moves. A single reversal is not counted.
`loop_step_fraction` divides flagged steps by all steps; `episodes_with_loops`
counts episodes with at least one flagged continuation. Ghosts can make a
reversal useful, so these are behavior diagnostics, not proof that every flagged
move is a policy mistake. Also compare score, win rate, survival length and food
progress; dying early must not be mistaken for solving oscillation.

### Navigation-feature results

Measured on the same 300 episode seeds, 40000-40299, with two ghosts on the
10 x 10 map (200-step limit):

| Metric | Original DQN | DQN with navigation features |
| --- | ---: | ---: |
| Mean score | 197.80 | 495.10 |
| Win rate | 0% | 56.67% |
| Mean episode length | 29.31 | 65.35 |
| Mean flagged loop steps per episode | 3.29 | 0.40 |
| Flagged loop steps / all steps | 11.24% | 0.61% |
| Episodes with any flagged loop | 47 / 300 | 28 / 300 |

The new run trained for 6000 episodes; validation selected episode 4500. The
original run trained for 12000 episodes and selected episode 5000. Both used
training seed 0, standard DQN targets and the same original game rewards. The
new epsilon decay spans 80000 environment steps (originally 100000), so this is
a before/after improvement, not a controlled feature-only ablation. Learning
across multiple seeds and maze layouts still needs evaluation.

For the original reported demonstration seed 0, the new policy scored 550 in
63 steps, with no flagged two-cell loops and at most two consecutive foodless
moves; it eventually died. This illustrates the behavioral improvement without
claiming every episode is won. Detailed metrics and all settings are recorded in
`dqn-oscillation-validation.json`; per-episode traces remain in local outputs.
All 43 regression tests passed, including the original 35 and eight new tests.

Use this **new checkpoint**, after activating the virtual environment:

```bash
python cli.py controller=dqn controller.mode=load \
  controller.model_path=outputs/dqn/navigation-seed0/best.pt \
  environment=ghosts eval=false framerate=8
```

The updated demonstration GIF is `outputs/dqn/navigation-seed0/gameplay.gif`.
