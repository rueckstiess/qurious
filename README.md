# qurious - A Minimalistic Reinforcement Learning Framework

This project aims to create a clean, extensible, and educational reinforcement learning (RL) framework in Python. The design emphasizes clarity, modularity, and a solid foundation for implementing classic RL algorithms.

## Project Structure

The framework consists of the following core components:

### Markov Decision Process (MDP)
- Represents the mathematical model of a decision-making problem
- Manages transition probabilities, rewards, discount factors, and terminal states
- Provides methods for sampling from the environment and calculating expected rewards

### Markov Reward Process (MRP)
- Represents an MDP with a fixed policy
- Can be derived from an MDP and a policy
- Provides methods for calculating state values based on the Bellman equation

### Policy
- Hierarchy of policy classes that map states to actions
- Abstract base class that defines the core policy interface
- Implementations include:
  - Deterministic tabular policies
  - Stochastic tabular policies
  - Epsilon-greedy exploration wrapper
  - Softmax policy with temperature parameter

### Value Functions
- Abstractions for state value (V) and action value (Q) functions
- Tabular implementations with methods for estimation and updates
- Support for calculating best actions and saving/loading value functions

### Agents
- Abstract Agent base class defining the core agent interface and supporting experience tracking
- Value-based agents including:
  - QLearningAgent (off-policy TD control)
  - SarsaAgent (on-policy TD control)
  - ExpectedSarsaAgent (hybrid approach)

### Environments
- GridWorld environment supporting:
  - Obstacles and goals
  - Customizable rewards
  - Conversion to MDP representation
  - Visual rendering

### Visualization
- Grid world visualization with policy/value overlays
- Learning curve plotting
- Textual policy display


## Design Principles

1. **Clean interfaces**: Abstract base classes define clear contracts for all components
2. **Extensibility**: Design patterns that allow for easy extension
3. **Numpy-based**: Efficient implementations using NumPy arrays
4. **Test-driven**: Comprehensive unit tests for all components
5. **Educational**: Clear implementations of core RL concepts

## Current Status

The framework currently supports tabular reinforcement learning methods, including:
- Dynamic Programming (through MDP/MRP implementations)
- Temporal Difference learning (Q-learning, SARSA, Expected SARSA)
- Basic environments (Grid World)
- Visualization tools for policies and learning progress

## Future Extensions

The framework is designed to accommodate more advanced features in the future:

- Monte Carlo methods
- TD(λ) with eligibility traces
- Additional environments (e.g., Windy Gridworld, Blackjack)
- Function approximation for policies and value functions (linear, neural networks)
- Advanced exploration strategies
- Multi-agent reinforcement learning
- Parallel/distributed training
- Deep RL extensions