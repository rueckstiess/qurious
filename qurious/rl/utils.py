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
            agent.learn()

            state = next_state

            if step_callback:
                step_callback(env, agent, transition=agent.experience.get_current_transition())

        if episode_callback:
            episode_callback(env, agent, episode=agent.experience.get_current_episode())

        agent.policy.decay_epsilon()


def run_agent(env, agent, num_episodes=100, max_steps_per_ep=None, step_callback=None, episode_callback=None):
    """
    Run an agent in an environment for a number of episodes.

        :param env: The environment to run the agent in.
        :param agent: The agent to run.
        :param num_episodes: The number of episodes to run the agent for.
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
            # Increment step counter to enforce max_steps_per_ep limit
            step_count += 1

            if step_callback:
                step_callback(env, agent, transition=agent.experience.get_current_transition())

        if episode_callback:
            episode_callback(env, agent, episode=agent.experience.get_current_episode())
