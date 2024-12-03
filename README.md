# **Tob the bot** - Reinforcement learning project

This is a game controlled by a reinforcement learning agent. The goal of the game is to collecting the rewards while navigate through and avoiding obstacles.

The agent plays a randomly generated level over and over again until it figures it out.
It's trained using a deep q-learning method with a greedy epsilon algorithm for exploration.

## Random level

![Example Gridworld Visualization](https://github.com/TheRomanOne/Reinforcement-Learning/blob/master/example.png?raw=true)

- **Left:** A randomly generated level, where the agent needs to move through free spaces (green) and obstacles (brown) to collect rewards (pile of gold) - endlessly.
- **Right:** A performance graph showing the agent's learning progress:
  - The **green line** number of steps as a function of epsilon (uncertainty). It represents the number of steps the agent took to finish the level within a given limit.
  - **Red dots** indicate when the target network was updated.

## Features
- Dynamic gridworld generation with customizable parameters.
- Deep Q-learning algorithm implementation with experience replay and target network updates.
- Visualization of agent's environment and learning process.

