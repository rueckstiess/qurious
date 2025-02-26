class Environment:
    def __init__(self):
        """Initialize the environment."""
        self._state = None
        self._done = False
        self._info = {}

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            state: The initial state after reset
        """
        self._done = False
        self._info = {}
        return self.get_state()

    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            state: Current state
        """
        return self._state

    def step(self, action):
        """
        Take an action in the environment and observe the outcome.

        Args:
            action: The action to take

        Returns:
            next_state: The new state after taking the action
            reward: The reward received
            done: Whether the episode has terminated
            info: Additional information (optional)
        """
        raise NotImplementedError("Subclasses must implement the step method")

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode: The rendering mode (e.g., 'human', 'rgb_array')

        Returns:
            Rendering based on the specified mode
        """
        raise NotImplementedError("Subclasses must implement the render method")

    def close(self):
        """Clean up resources."""
        pass

    def sample_action(self):
        """
        Sample a random action from the action space.

        Returns:
            action: A sampled action
        """
        raise NotImplementedError("Subclasses must implement the sample_action method")

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space
        """
        raise NotImplementedError("Subclasses must implement the action_space property")

    @property
    def observation_space(self):
        """
        Get the observation space of the environment.

        Returns:
            The observation space
        """
        raise NotImplementedError("Subclasses must implement the observation_space property")

    @property
    def done(self):
        """
        Check if the episode is done.

        Returns:
            done: Whether the episode has terminated
        """
        return self._done
