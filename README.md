# Reinforcement Learning: Gridworld Agent

This repository contains an implementation of a reinforcement learning agent navigating a gridworld environment. The agent is trained to move through the grid while avoiding obstacles and collecting rewards.

## Example Visualization

![Example Gridworld Visualization](https://github.com/TheRomanOne/Reinforcement-Learning/blob/master/example.png?raw=true)

- **Left:** The gridworld environment, where the agent (robot icon) navigates through free spaces (green) and obstacles (brown) to collect rewards (pile of gold).
- **Right:** A performance graph showing the agent's learning progress:
  - The **green line** represents the number of steps per episode as epsilon (exploration rate) decreases.
  - **Red dots** indicate when the target network was updated.

## Features
- Dynamic gridworld generation with customizable parameters.
- Deep Q-learning algorithm implementation with experience replay and target network updates.
- Visualization of agent's environment and learning process.

