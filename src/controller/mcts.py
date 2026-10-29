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
    __slots__ = ("actions", "unexpanded", "visits")

    def __init__(self, actions):
        self.actions = {}
        self.unexpanded = list(actions)
        self.visits = 0


class MCTSAgent(Controller):
    """Plan online with UCT and random rollouts from the current game state.

    The environment is copied for every simulation, so search does not mutate
    the live game. Ghost moves remain stochastic and are sampled by the
    environment's normal ``step`` implementation.
    """

    def __init__(self, environment, iterations=250, exploration=1.4,
                 max_depth=30, gamma=0.98, seed=0):
        if iterations <= 0 or max_depth <= 0:
            raise ValueError("iterations and max_depth must be positive")
        if exploration < 0 or not 0 <= gamma <= 1:
            raise ValueError("exploration must be non-negative and gamma in [0, 1]")
        self.environment = environment
        self.iterations = int(iterations)
        self.exploration = float(exploration)
        self.max_depth = int(max_depth)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)

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
        return max(
            node.actions.items(),
            key=lambda item: (
                item[1].value_sum / item[1].visits
                + self.exploration * math.sqrt(log_parent / item[1].visits)
            ),
        )

    def _rollout(self, environment, depth):
        rewards = []
        while not environment.done and depth < self.max_depth:
            actions = self._legal_actions(environment)
            if not actions:
                break
            observation = environment.step(self.rng.choice(actions))
            rewards.append(observation.reward)
            depth += 1
        return rewards

    def _simulate(self, tree, root_key):
        environment = deepcopy(self.environment)
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
                tree.setdefault(child_key, _Node(self._legal_actions(environment)))
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
            tree.setdefault(node_key, _Node(self._legal_actions(environment)))

        rewards.extend(self._rollout(environment, depth))
        for start, (node, edge) in enumerate(path):
            value = 0.0
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
        tree = {root_key: _Node(legal)}
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
