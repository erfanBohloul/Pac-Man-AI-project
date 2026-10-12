"""Deep Q-learning on structured Pac-Man states (no screen capture required)."""
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
import json
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.controller.controller import Controller
from src.state import ActionSpaceEnum, GhostColorEnum, Observation, Position


@dataclass
class DQNConfig:
    hidden_size: int = 128
    learning_rate: float = 0.0005
    gamma: float = 0.98
    batch_size: int = 64
    replay_capacity: int = 20000
    learning_starts: int = 1000
    train_every: int = 4
    target_update_every: int = 250  # Optimizer updates, not environment steps.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    reward_scale: float = 0.1
    grad_clip: float = 10.0
    double_dqn: bool = False
    mask_actions: bool = True
    navigation_features: bool = False
    wall_features: bool = True
    local_wall_features: bool = False
    seed: int = 0
    device: str = "cpu"  # Small MLPs are usually faster on CPU for this game.
    num_threads: int = 1

    def __post_init__(self):
        for name in ("hidden_size", "batch_size", "replay_capacity", "train_every",
                     "target_update_every", "epsilon_decay_steps", "num_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size > self.replay_capacity or self.learning_starts < 0:
            raise ValueError("Invalid replay capacity, batch size or learning_starts")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid epsilon range or discount factor")
        if min(self.learning_rate, self.reward_scale, self.grad_clip) <= 0:
            raise ValueError("Learning rate, reward scale and gradient clip must be positive")


class StateEncoder:
    VERSION = 1

    def __init__(self, grid_size, max_steps, stability_rate=0.0, navigation_features=False,
                 wall_features=True, local_wall_features=False):
        if grid_size < 3 or max_steps <= 0:
            raise ValueError("Invalid grid size or episode limit")
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.stability_rate = stability_rate
        self.navigation_features = navigation_features
        self.wall_features = wall_features
        self.local_wall_features = local_wall_features
        self.input_size = ((2 + int(wall_features)) * grid_size * grid_size + 6 * 8 + 2
                           + (12 if navigation_features else 0) + (4 if local_wall_features else 0))

    @staticmethod
    def local_walls(state):
        """Explicit wall bits in ActionSpaceEnum order: up, right, down, left."""
        return np.array([
            state._state_after_action(action) in state.walls
            for action in ActionSpaceEnum
        ], dtype=np.float32)

    @staticmethod
    def distance_map(state, sources):
        """Shortest walk distances to visible targets, respecting walls."""
        distances = {position: 0 for position in sources if position not in state.walls}
        queue = deque(distances)
        while queue:
            position = queue.popleft()
            for dx, dy in state.directions.values():
                neighbor = Position(position.x + dx, position.y + dy)
                if (0 <= neighbor.x < state.size and 0 <= neighbor.y < state.size and
                        neighbor not in state.walls and neighbor not in distances):
                    distances[neighbor] = distances[position] + 1
                    queue.append(neighbor)
        return distances

    def navigation(self, state):
        """Per action: immediate food, food proximity, ghost proximity.

        These are features of the observed map, not a prescribed move or a
        rollout. Missing/unreachable targets have zero proximity; walls use -1.
        """
        food = self.distance_map(state, state.pellets)
        ghosts = self.distance_map(state, state.ghost_positions)
        features = np.full((4, 3), -1.0, dtype=np.float32)
        for action in state.get_legal_actions():
            position = state._state_after_action(action)
            features[int(action)] = (
                float(position in state.pellets),
                1 / (1 + food[position]) if position in food else 0,
                1 / (1 + ghosts[position]) if position in ghosts else 0,
            )
        return features.ravel()

    def __call__(self, observation: Observation):
        state = observation.map
        if state.size != self.grid_size:
            raise ValueError("Checkpoint and environment grid sizes differ")
        # Separate layers retain food underneath ghosts and Pac-Man.
        board = np.zeros((3, state.size, state.size), dtype=np.float32)
        for position in state.walls:
            board[0, position.y, position.x] = 1
        for position in state.pellets:
            board[1, position.y, position.x] = 1
        board[2, state.pacman_position.y, state.pacman_position.x] = 1
        # One slot per color: present, normalized x/y, one-hot direction incl. None.
        ghosts = np.zeros((6, 8), dtype=np.float32)
        colors = {color: i for i, color in enumerate(GhostColorEnum)}
        for position in state.ghost_positions:
            color = state.ghost_position_to_color[position]
            slot = ghosts[colors[color]]
            slot[:3] = (1, position.x / (state.size - 1), position.y / (state.size - 1))
            direction = state.ghost_position_to_prev_action.get(position)
            slot[3 + (4 if direction is None else int(direction))] = 1
        time_left = max(0.0, 1 - observation.step_count / self.max_steps)
        # On a fixed maze the wall plane is constant; omit it for an ablation.
        board_features = board if self.wall_features else board[1:]
        parts = [board_features.ravel(), ghosts.ravel(),
                 np.array([time_left, self.stability_rate], dtype=np.float32)]
        if self.navigation_features:
            parts.append(self.navigation(state))
        if self.local_wall_features:
            parts.append(self.local_walls(state))
        return np.concatenate(parts)


class QNetwork(nn.Sequential):
    def __init__(self, input_size, hidden_size):
        super().__init__(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.cursor = 0

    def __len__(self):
        return len(self.items)

    def add(self, state, action, reward, next_state, done, next_mask):
        # Own the arrays: environment observations themselves are mutable.
        transition = (state.copy(), int(action), float(reward), next_state.copy(),
                      bool(done), next_mask.copy())
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, size, rng):
        rows = rng.sample(self.items, size)
        return tuple(np.stack(column) for column in zip(*rows))


def bellman_targets(rewards, dones, next_values, next_masks, gamma, online_next=None):
    """Terminal and no-action states bootstrap to zero; ignore wall actions."""
    masked = next_values.masked_fill(~next_masks, -torch.inf)
    if online_next is None:
        values = masked.max(dim=1).values
    else:
        actions = online_next.masked_fill(~next_masks, -torch.inf).argmax(dim=1)
        values = masked.gather(1, actions.unsqueeze(1)).squeeze(1)
    values = torch.where(dones | ~next_masks.any(dim=1), 0.0, values)
    return rewards + gamma * values


def environment_spec(env):
    from src.environment import GhostsPacmanEnvironment
    spec = {"environment_name": "ghosts" if isinstance(env, GhostsPacmanEnvironment) else "basic",
            "grid_size": env.grid_size, "max_steps": env.max_steps}
    if isinstance(env, GhostsPacmanEnvironment):
        spec.update(num_ghosts=env.num_ghosts, stability_rate=env.ghosts_strategy.stability_rate)
    return spec


class DQNAgent(Controller):
    CHECKPOINT_VERSION = 1

    def __init__(self, environment, config=None):
        self.config = config or DQNConfig()
        cfg = self.config
        self.environment_spec = environment_spec(environment)
        self.encoder = StateEncoder(environment.grid_size, environment.max_steps,
                                    self.environment_spec.get("stability_rate", 0.0),
                                    cfg.navigation_features, cfg.wall_features, cfg.local_wall_features)
        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = cfg.device
        self.device = torch.device(device)
        torch.set_num_threads(cfg.num_threads)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.online = QNetwork(self.encoder.input_size, cfg.hidden_size).to(self.device)
        self.target = QNetwork(self.encoder.input_size, cfg.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.episodes = 0
        self.online.eval()

    @property
    def epsilon(self):
        ratio = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + ratio * (self.config.epsilon_end - self.config.epsilon_start)

    def action_mask(self, observation):
        mask = np.ones(4, dtype=bool)
        if self.config.mask_actions:
            mask[:] = False
            mask[list(observation.map.get_legal_actions())] = True
        return mask

    def q_values(self, observation):
        """Raw learned values in scaled reward units, useful for later MCTS."""
        with torch.no_grad():
            state = torch.from_numpy(self.encoder(observation)).to(self.device).unsqueeze(0)
            return self.online(state)[0].cpu().numpy()

    def select_action(self, observation, epsilon=0.0):
        valid = np.flatnonzero(self.action_mask(observation))
        if not len(valid):
            # A trapped agent can still spend a step; all four moves hit walls.
            return ActionSpaceEnum.UP
        if epsilon > 0 and self.rng.random() < epsilon:
            return ActionSpaceEnum(int(self.rng.choice(valid)))
        values = self.q_values(observation)
        return ActionSpaceEnum(int(valid[np.argmax(values[valid])]))

    def get_action(self, observation):
        return self.select_action(observation)  # Always greedy during evaluation/display.

    def optimize(self):
        cfg = self.config
        if len(self.replay) < cfg.batch_size:
            return None
        state, action, reward, next_state, done, masks = self.replay.sample(cfg.batch_size, self.rng)
        tensors = [torch.as_tensor(x, device=self.device) for x in
                   (state, action, reward, next_state, done, masks)]
        state, action, reward, next_state, done, masks = tensors
        reward = reward.float() * cfg.reward_scale
        predicted = self.online(state).gather(1, action.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = bellman_targets(reward, done, self.target(next_state), masks, cfg.gamma,
                                     self.online(next_state) if cfg.double_dqn else None)
        loss = F.smooth_l1_loss(predicted, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss became non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.CHECKPOINT_VERSION, "encoder_version": StateEncoder.VERSION,
            "config": asdict(self.config), "environment": self.environment_spec,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
            "episodes": self.episodes,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, environment, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("version") != cls.CHECKPOINT_VERSION or payload.get("encoder_version") != StateEncoder.VERSION:
            raise ValueError("Unsupported DQN checkpoint or state encoder version")
        if payload["environment"] != environment_spec(environment):
            raise ValueError(f"Checkpoint environment mismatch: expected {payload['environment']}, "
                             f"received {environment_spec(environment)}")
        agent = cls(environment, DQNConfig(**{**payload["config"], "device": device}))
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        for name in ("environment_steps", "gradient_steps", "episodes"):
            setattr(agent, name, payload[name])
        return agent

    def train(self, env, episodes, output_dir, eval_every=100, eval_episodes=20, eval_seed=1000000):
        """Train from interactions, validate on fixed seeds, save best and last policies.

        Checkpoints are for inference/warm starts; replay and optimizer state are
        not serialized, so exact interrupted-training resumption is not supported.
        """
        from copy import deepcopy
        from src.evaluation import evaluate_algorithm
        if episodes <= 0 or eval_every <= 0 or eval_episodes <= 0:
            raise ValueError("Training and evaluation episode counts must be positive")
        if max(self.config.seed, eval_seed) < min(self.config.seed + episodes, eval_seed + eval_episodes):
            raise ValueError("Training and validation seed ranges must not overlap")
        if environment_spec(env) != self.environment_spec:
            raise ValueError("Training environment differs from the agent's encoder settings")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "training.jsonl").exists():
            raise FileExistsError(f"Use a fresh output directory to preserve previous runs: {output}")
        validation_env = deepcopy(env)
        run = {"config": asdict(self.config), "environment": self.environment_spec,
               "episodes": episodes, "eval_seed": eval_seed, "eval_episodes": eval_episodes}
        (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        best_reward = -float("inf")
        started = perf_counter()
        validation_seconds = 0.0
        with (output / "training.jsonl").open("w") as log:
            for episode in range(episodes):
                random.seed(self.config.seed + episode)
                observation = env.reset()
                total_reward, losses = 0.0, []
                self.online.train()
                while not observation.done:
                    state = self.encoder(observation)  # Capture BEFORE step mutates the map.
                    action = self.select_action(observation, self.epsilon)
                    observation = env.step(action)
                    self.replay.add(state, action, observation.reward, self.encoder(observation),
                                    observation.done, self.action_mask(observation))
                    self.environment_steps += 1
                    total_reward += observation.reward
                    if (self.environment_steps >= self.config.learning_starts and
                            self.environment_steps % self.config.train_every == 0):
                        loss = self.optimize()
                        if loss is not None:
                            losses.append(loss)
                self.episodes += 1
                row = {"episode": self.episodes, "steps": observation.step_count,
                       "environment_steps": self.environment_steps, "score": observation.score,
                       "reward": total_reward, "epsilon": self.epsilon,
                       "loss": float(np.mean(losses)) if losses else None}
                if self.episodes % eval_every == 0 or episode == episodes - 1:
                    self.online.eval()
                    validation_started = perf_counter()
                    metrics = evaluate_algorithm(validation_env, self, eval_episodes, seed=eval_seed)
                    validation_seconds += perf_counter() - validation_started
                    row["validation"] = metrics
                    if metrics["Mean-reward"] > best_reward:
                        best_reward = metrics["Mean-reward"]
                        self.save_model(output / "best.pt")
                    self.save_model(output / "last.pt")
                    print(f"Episode {self.episodes}/{episodes} | steps {self.environment_steps} | "
                          f"epsilon {self.epsilon:.3f} | validation score {metrics['Mean-score']:.1f} | "
                          f"reward {metrics['Mean-reward']:.1f} | wins {metrics['Win-rate']:.0%}", flush=True)
                row['elapsed_seconds'] = perf_counter() - started
                log.write(json.dumps(row) + "\n")
                log.flush()
        self.online.eval()
        (output / 'timing.json').write_text(json.dumps({
            'total_seconds': perf_counter() - started,
            'validation_seconds': validation_seconds,
            'environment_steps': self.environment_steps,
            'episodes': episodes,
            'device': str(self.device),
            'num_threads': self.config.num_threads,
            'scope': 'Training loop including periodic validation and checkpoint/log writes; excludes setup and final test evaluation.',
        }, indent=2) + '\n')
        return output / "best.pt"
