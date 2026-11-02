import random
from src.controller import Controller
from src.state import Observation, ActionSpaceEnum


class BasicController(Controller):
    """
    A basic controller that selects actions at random.

    This controller is useful as a baseline or for testing the environment.
    It uniformly chooses one of the available actions without considering
    the provided observation.
    """

    def __init__(self, legal_only=False, seed=None):
        """
        Initializes the BasicController with all possible actions.

        The action space includes four discrete actions: UP, RIGHT, DOWN, and LEFT.
        """
        self.legal_only = legal_only
        self.rng = random.Random(seed) if seed is not None else random
        self.action_space = [
            ActionSpaceEnum.UP,
            ActionSpaceEnum.RIGHT,
            ActionSpaceEnum.DOWN,
            ActionSpaceEnum.LEFT
        ]

    def get_action(self, observation: Observation) -> ActionSpaceEnum:
        """
        Randomly selects an action from the action space.

        Args:
            observation (Observation): The current state of the environment. 
                                       This implementation does not use the observation 
                                       to make decisions.

        Returns:
            ActionSpaceEnum: A randomly chosen action from the action space.
        """
        actions = observation.map.get_legal_actions() if self.legal_only else self.action_space
        return self.rng.choice(actions or self.action_space)
