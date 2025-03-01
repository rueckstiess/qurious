import os


def train_agent(env, agent, num_episodes=100, step_callback=None, episode_callback=None):
    """
    Train an agent in an environment for a number of episodes.

        :param env: The environment to train the agent in.
        :param agent: The agent to train.
        :param num_episodes: The number of episodes to train the agent for.
        :param step_callback: A callback function to call after each step.
        :param episode_callback: A callback function to call after each episode.

    """
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn(agent.experience.get_current_transition())
            state = next_state

            if step_callback:
                step_callback(env, agent, transition=agent.experience.get_current_transition())

        if episode_callback:
            episode_callback(env, agent, episode=agent.experience.get_current_episode())

        agent.policy.decay_epsilon()


def run_agent(env, agent, num_episodes=100, max_steps_per_ep=None, step_callback=None, episode_callback=None):
    """
    Train an agent in an environment for a number of episodes.

        :param env: The environment to train the agent in.
        :param agent: The agent to train.
        :param num_episodes: The number of episodes to train the agent for.
        :param max_steps_per_ep: The maximum number of steps per episode.
        :param step_callback: A callback function to call after each step.
        :param episode_callback: A callback function to call after each episode.

    """
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        while not done and (max_steps_per_ep is None or step_count < max_steps_per_ep):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if step_callback:
                step_callback(env, agent, transition=agent.experience.get_current_transition())

        if episode_callback:
            episode_callback(env, agent, episode=agent.experience.get_current_episode())


def clear_output():
    """
    Detects whether code is running in a Jupyter notebook or terminal
    and clears the output accordingly.

    Returns:
        bool: True if running in Jupyter, False if in terminal
    """
    # Try to detect if we're in a Jupyter environment
    try:
        # This will only work in IPython/Jupyter environments
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is not None and "IPKernelApp" in ipython.config:
            # We're in Jupyter notebook or qtconsole
            from IPython.display import clear_output as jupyter_clear

            jupyter_clear(wait=True)
            return True
        else:
            # We're in terminal IPython or standard Python
            os.system("cls" if os.name == "nt" else "clear")
            return False
    except (ImportError, NameError):
        # We're in standard Python
        os.system("cls" if os.name == "nt" else "clear")
        return False
