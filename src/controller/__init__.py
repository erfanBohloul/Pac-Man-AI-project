from .controller import Controller
from .basic import BasicController
from .qlearn import QTable, QLearnAgent
from .value_iteration import ValueIterationAgent
from .dqn import DQNAgent, DQNConfig


def __getattr__(name):
    # Legacy policy-gradient dependencies are only loaded when requested.
    if name in {"ReinforceTrainer", "NeuralNetworkPolicy"}:
        from . import reinforce
        return getattr(reinforce, name)
    raise AttributeError(name)
