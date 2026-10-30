# Pac-Man Reinforcement Learning Project

A modular framework for experimenting with reinforcement learning algorithms by training an agent to control Pac-Man in a simplified grid-based environment.

---

## Description
This project implements a simplified version of the classic Pac-Man game, designed as an environment for experimenting with reinforcement learning (RL) algorithms. The goal is to train an RL agent to control Pac-Man, navigating a grid-based maze, collecting pellets, avoiding ghosts, and maximizing the score.

The project demonstrates the power of RL techniques by providing a modular framework where various RL algorithms can be applied.

---

## MDP Problem Definition

We model the Pac-Man game as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$, where:

### State Space $\mathcal{S}$

The state space is represented by a matrix $S \in \mathbb{R}^{n \times n}$, where $n$ is the grid size. Each element $s_{ij}$ of the matrix can take the following values:

$$
s_{ij} = \begin{cases}
0 & \text{empty cell} \\
1 & \text{pellet} \\
-1 & \text{wall} \\
2 & \text{Pac-Man position} \\
-2 & \text{ghost position}
\end{cases}
$$

`Map.to_numpy()` returns this grid flattened, with each position `(x, y)`
written to row `y`, column `x`. This grid is an observation encoding, not the
complete simulator state: ghost directions are stored separately in
`map.ghost_position_to_prev_action`, and pellets under ghosts remain in
`map.pellets`. The DQN encoder includes those features and the remaining episode
time in separate inputs; see [the DQN guide](docs/dqn.md).

### Action Space $\mathcal{A}$

The action space is defined as:

$$
\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}
$$

Mathematically, we can represent these actions as unit vectors:

$$
\text{Up} = (-1, 0), \text{Down} = (1, 0), \text{Left} = (0, -1), \text{Right} = (0, 1)
$$

### Reward Function $\mathcal{R}$

The reward function $$\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$ is defined as:

$$
\mathcal{R}(s, a, s') = \begin{cases}
0 & \text{if the episode was already finished} \\
-100 & \text{if Pac-Man collides with a ghost} \\
-50 & \text{if Pac-Man hits a wall without dying} \\
9 & \text{if Pac-Man eats a pellet without dying} \\
-1 & \text{otherwise}
\end{cases}
$$

Eating a pellet increases the score by 10; its reward includes the -1 step
cost. Collision is checked both before and after ghost movement and overrides
the other rewards. Ghosts do not move if Pac-Man has already stepped onto one.

### Transition Model $\mathcal{P}$

The transition probability $\mathcal{P}(s'|s,a)$ is defined as follows:

For Pac-Man:

$$
\mathcal{P}(s'|s, a) = \begin{cases}
1 & \text{if } s' \text{ is the deterministic result of action} a \text{ in state } s \\
0 & \text{ otherwise }
\end{cases}
$$

For ghosts:
Let $d$ be the previous direction, $D$ the available directions excluding walls
and other ghosts, and $p$ the configured `stability_rate` (default 0.75).
For nonempty $D$:

$$
\mathcal{P}(d'|d) = \begin{cases}
p + \frac{1-p}{|D|} & \text{if } d' = d \text{ and } d \in D \\
\frac{1-p}{|D|} & \text{if } d' \neq d \text{ and } d, d' \in D \\
\frac{1}{|D|} & \text{if } d \notin D \text{ and } d' \in D \\
0 & \text{if } d' \notin D
\end{cases}
$$

An uninitialized direction is treated as $d \notin D$. A ghost with no available
move stays in place. Ghosts move sequentially in color-name order, including in
reconstructed simulations.

### Finish Conditions

The game terminates when:
1. Pac-Man collects all pellets (`map.pellets` is empty).
2. Pac-Man occupies the same cell as a ghost, before or after ghost movement.
3. The episode reaches `max_steps` (default 200).

## Simulation and state snapshots

Use `env.clone()` on an initialized `GhostsPacmanEnvironment` to copy the map,
ghost directions, score, elapsed steps, episode limit, terminal flag and ghost
stability rate. The copied game can be stepped without changing the source map
or ghosts. `Map.clone()` also preserves the `MapFullHash` subclass.

```python
simulation = env.clone()
result = simulation.step(action)
```

To build a simulation from a saved map, supply the episode settings explicitly:

```python
simulation = GhostsPacmanEnvironment.from_map(
    saved_map,
    past_steps=100,
    max_steps=200,  # Total episode limit: 100 steps remain.
    score=score,
    stability_rate=0.75,
)
```

Terminal states can also be reconstructed; stepping them returns `done=True`
and zero reward without changing the state. Maps created without direction
metadata start ghosts with no previous direction.

Observations currently refer to the live map. For a replay buffer, capture
`observation.map.clone()` or encode the state **before** the next environment
step. Ghost sampling still uses Python's global `random` generator; cloning
does not create a separate random stream.

Full-state Q-table keys now include ghost identities and directions. Retrain
existing full-state checkpoints; their integer keys use the old encoding.
Historical scores in the controller documentation predate these environment
fixes and should be remeasured for algorithm comparisons.

Run the environment regression tests with Python and NumPy (no GUI or training
libraries required):

```bash
python -m unittest discover -s test -p '*_test.py' -v
```

---

## Build and Run

For the new DQN implementation, training, saved models, benchmarks and visual
playback, see [the DQN guide](docs/dqn.md).

An online Monte Carlo Tree Search controller is also available with the
`controller=mcts` preset. It uses UCT selection and random rollouts from copied
environment states; adjust its search budget with `controller.iterations` and
`controller.max_depth`. For example, launch it in the visual ghosts environment
with:

```bash
python cli.py controller=mcts environment=ghosts eval=false
```

The `controller=dqn_mcts` preset combines a trained DQN checkpoint with MCTS.
The DQN supplies action priors, guides rollout actions, and estimates value at
the search depth limit. The default preset loads the included local-wall DQN
checkpoint, which expects the default 10x10 ghosts environment. Override
`controller.model_path` to use another compatible DQN checkpoint. For example:

```bash
python cli.py controller=dqn_mcts environment=ghosts eval=false
```

The `controller=dqn_local` preset uses four adjacent-wall indicators in place
of the full wall plane (`266 -> 128 -> 128 -> 4` on the default 10 x 10 maze).
See [the local-wall experiment](docs/dqn-local-walls.md) for training, visual
playback and the matched-seed comparison.

1. Clone this repository:
   ```bash
   git clone --branch DQN https://github.com/erfanBohloul/Pac-Man-AI-project.git
   cd Pac-Man-AI-project
   ```
2. Install the dependencies:
   ```bash
   pip install .
   ```
3. To run the game with different configurations:
   ```bash
   python cli.py controller=random environment=basic
   python cli.py controller=qlearn environment=ghosts
   python cli.py controller=value_iteration environment=basic
   ```

4. To load and run a previously trained model:
   ```bash
   python cli.py controller=qlearn controller.qlearn.model_path=checkpoints/qlearn//checkpoint.pkl environment=ghosts
   ```

Note: When using model_path:
- If the file exists, the model will be loaded from it
- If the path doesn't exist or is omitted, the model will be trained from scratch

---

## Demonstration
Below is a demonstration of the project in action:

### Random walk policy:
![Pac-Man Gameplay random](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/random.gif)

### Q-learning (SARSA update):
![Pac-Man Gameplay Q-learning](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/q-learning.gif)

### Value Iteration
![Pac-Man Gameplay Value Iteration](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/value_iteration.gif)

### Cross Entropy
![Pac-Man Gameplay Cross Entropy](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/cross_entropy.gif)

### REINFORCE (Basic Environment, legacy demo)
![Pac-Man Gameplay REINFORCE (Basic Environment)](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/reinforce.gif)

### REINFORCE (Ghosts Environment, legacy demo)
![Pac-Man Gameplay REINFORCE (Ghosts Environment)](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/reinforce_ghosts.gif)


---

## Application Architecture

The project is structured into modular components, making it easy to extend or replace individual parts.

### Core Components
1. **Environment (`environment`)**:
   - Defines the game world, including walls, pellets, ghosts, and Pac-Man's position.
   - Handles game logic such as movement, collision detection, and reward calculation.
   - Implementations: `BasicPacmanEnvironment` and `GhostsPacmanEnvironment`

2. **Controller (`controller`)**:
   - Decides Pac-Man's actions based on observations from the environment.
   - Supports different controllers:
     - `BasicController`: Selects random actions for testing.
     - `QLearnAgent`: Implements Q-learning algorithm.
     - `ValueIterationAgent`: Implements Value Iteration algorithm.

3. **Drawer (`drawer`)**:
   - Visualizes the game state using Pygame.
   - `PygameDrawer` renders walls, pellets, ghosts, and Pac-Man on a grid.

4. **State (`state`)**:
   - Encapsulates the observation data passed from the environment to the controller.
   - Includes positions of walls, pellets, ghosts, and Pac-Man.

### Workflow
1. The environment generates an observation of the current game state.
2. The controller receives this observation and selects an action.
3. The environment processes this action, updates the game state, and calculates rewards.
4. The drawer visualizes the updated state.


# Authors
1. [Aleksander Ploskin](https://github.com/Alexander-Ploskin)
2. [Danil Tokhchukov](https://github.com/makriot)
3. [Sameer Tantry](https://github.com/sameertantry)
4. [Vladislav Gromadskii](https://github.com/WladGrm)
