"""Monte Carlo Tree Search controller for the Pac-Man environments."""

from copy import deepcopy
import math
import random

from src.controller.controller import Controller
from src.state import ActionSpaceEnum, Observation


class _Edge:
    """Statistics for one action edge in the search tree."""

    __slots__ = ("visits", "value_sum")

    def __init__(self):
        self.visits = 0
        self.value_sum = 0.0


class _Node:
    __slots__ = ("actions", "unexpanded", "visits", "priors")

    def __init__(self, actions, priors=None):
        self.actions = {}
        self.unexpanded = list(actions)
        self.visits = 0
        self.priors = priors or {action: 1 / len(actions) for action in actions} if actions else {}


class MCTSAgent(Controller):
    """Plan online with UCT, optionally guided by a trained DQN policy/value.

    The environment is copied for every simulation, so search does not mutate
    the live game. Ghost moves remain stochastic and are sampled by the
    environment's normal ``step`` implementation. With ``dqn_agent`` set,
    its Q-values provide PUCT action priors, guide rollout actions, and
    bootstrap value at the search depth limit.
    """

    def __init__(self, environment, iterations=250, exploration=1.4,
                 max_depth=30, gamma=0.98, seed=0, dqn_agent=None,
                 prior_temperature=1.0, rollout_epsilon=0.1):
        if iterations <= 0 or max_depth <= 0:
            raise ValueError("iterations and max_depth must be positive")
        if exploration < 0 or not 0 <= gamma <= 1:
            raise ValueError("exploration must be non-negative and gamma in [0, 1]")
        if prior_temperature <= 0 or not 0 <= rollout_epsilon <= 1:
            raise ValueError("prior_temperature must be positive and rollout_epsilon in [0, 1]")
        self.environment = environment
        self.iterations = int(iterations)
        self.exploration = float(exploration)
        self.max_depth = int(max_depth)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        self.dqn_agent = dqn_agent
        self.prior_temperature = float(prior_temperature)
        self.rollout_epsilon = float(rollout_epsilon)

    def _make_node(self, actions, observation):
        if not actions:
            return _Node(actions)
        if self.dqn_agent is None:
            return _Node(actions)
        import numpy as np
        values = self.dqn_agent.q_values(observation)[[int(action) for action in actions]]
        values = values - np.max(values)
        weights = np.exp(values / self.prior_temperature)
        priors = {action: float(weight / weights.sum()) for action, weight in zip(actions, weights)}
        return _Node(actions, priors)

    @staticmethod
    def _state_key(environment):
        state = environment.map
        ghosts = tuple(sorted(
            (position.x, position.y,
             str(state.ghost_position_to_color.get(position, "")),
             (None if state.ghost_position_to_prev_action.get(position) is None
              else int(state.ghost_position_to_prev_action[position])))
            for position in state.ghost_positions
        ))
        return (
            state.pacman_position.x, state.pacman_position.y,
            tuple(sorted((p.x, p.y) for p in state.pellets)), ghosts,
            environment.step_count,
        )

    @staticmethod
    def _legal_actions(environment):
        return environment.map.get_legal_actions()

    def _select_edge(self, node):
        log_parent = math.log(max(1, node.visits))
        if self.dqn_agent is not None:
            return max(
                node.actions.items(),
                key=lambda item: (
                    item[1].value_sum / item[1].visits
                    + self.exploration * node.priors[item[0]]
                    * math.sqrt(node.visits) / (1 + item[1].visits)
                ),
            )
        return max(
            node.actions.items(),
            key=lambda item: (
                item[1].value_sum / item[1].visits
                + self.exploration * math.sqrt(log_parent / item[1].visits)
            ),
        )

    def _rollout(self, environment, observation, depth):
        rewards = []
        while not environment.done and depth < self.max_depth:
            actions = self._legal_actions(environment)
            if not actions:
                break
            if self.dqn_agent is not None and self.rng.random() >= self.rollout_epsilon:
                q_values = self.dqn_agent.q_values(observation)
                action = max(actions, key=lambda candidate: q_values[int(candidate)])
            else:
                action = self.rng.choice(actions)
            observation = environment.step(action)
            rewards.append(observation.reward)
            depth += 1
        bootstrap = 0.0
        if self.dqn_agent is not None and not environment.done and depth >= self.max_depth:
            # DQN values use reward_scale units; MCTS backs up environment rewards.
            scale = self.dqn_agent.config.reward_scale
            bootstrap = max(self.dqn_agent.q_values(observation)[[int(a) for a in self._legal_actions(environment)]],
                            default=0.0) / scale
        return rewards, bootstrap

    def _simulate(self, tree, root_key):
        environment = deepcopy(self.environment)
        observation = Observation(0, environment.done, environment.score,
                                 environment.step_count, environment.map)
        node_key = root_key
        path = []
        rewards = []
        depth = 0

        while depth < self.max_depth and not environment.done:
            node = tree[node_key]
            if node.unexpanded:
                action = self.rng.choice(node.unexpanded)
                node.unexpanded.remove(action)
                observation = environment.step(action)
                rewards.append(observation.reward)
                depth += 1
                child_key = self._state_key(environment)
                if child_key not in tree:
                    tree[child_key] = self._make_node(self._legal_actions(environment), observation)
                edge = _Edge()
                node.actions[action] = edge
                path.append((node, edge))
                node_key = child_key
                break

            if not node.actions:
                break
            action, edge = self._select_edge(node)
            observation = environment.step(action)
            rewards.append(observation.reward)
            depth += 1
            path.append((node, edge))
            node_key = self._state_key(environment)
            if node_key not in tree:
                tree[node_key] = self._make_node(self._legal_actions(environment), observation)

        rollout_rewards, bootstrap = self._rollout(environment, observation, depth)
        rewards.extend(rollout_rewards)
        for start, (node, edge) in enumerate(path):
            value = bootstrap
            for reward in reversed(rewards[start:]):
                value = reward + self.gamma * value
            edge.visits += 1
            edge.value_sum += value
            node.visits += 1

    def get_action(self, observation: Observation) -> ActionSpaceEnum:
        legal = self._legal_actions(self.environment)
        if not legal:
            # This can only occur for malformed or terminal observations; the
            # runner normally asks for an action only while the episode lives.
            return ActionSpaceEnum.UP

        root_key = self._state_key(self.environment)
        tree = {root_key: self._make_node(legal, observation)}
        for _ in range(self.iterations):
            self._simulate(tree, root_key)

        root = tree[root_key]
        if not root.actions:
            return self.rng.choice(legal)
        return max(
            root.actions.items(),
            key=lambda item: (
                item[1].visits,
                item[1].value_sum / item[1].visits,
            ),
        )[0]
