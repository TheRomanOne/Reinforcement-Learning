import os, warnings
import numpy as np
from NeuralAgent import DQLAgent
from Scene import Scene
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.system('clear')

"""
each reward has it's fixed position

while 0.65 < epsilon <0.9:
    the agent starts every time in a random position with a random subset of total rewards.
    this can help exploration and collect more efficient data by creating scenes where
    not all rewards are present and the player is in either position
else:
    player starts at the same position with all the rewards and needs to learn to navigate through them
"""

cell_size = 40
grid_shape = (7, 7)
MAX_STEPS = min(1000, 10 * grid_shape[0] * grid_shape[1])
window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)

use_convolution=True

scene = Scene(
    window_shape=window_shape, 
    grid_shape=grid_shape, 
    matrix_state=use_convolution, 
    # seed=1368,
    num_of_rewards=3,
    free_space_prob=.75
)

agent = DQLAgent(
    state_size=scene.get_state().shape,
    epsilon_decay=0.9973,
    gamma=.9,
    action_size=4,
    use_convolution=use_convolution
)

# episodes = 500
lowest_steps = 99999999

rewards = []
steps = []
epsilons = []
while agent.epsilon > 0.1:
    state = scene.reset(random_player_position=.65 < agent.epsilon < .9)
    total_reward = 0
    losses = []
    
    while len(scene.rewards) > 0:
        if scene.steps > MAX_STEPS:
            break
        action = agent.act(state)

        next_state, reward, done = scene.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train()
        if loss > -1:
            losses.append(loss)
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
    steps.append(scene.steps)
    epsilons.append(agent.epsilon)

    collected = scene.reward_batch_size - len(scene.rewards)
    print(f"Episode {scene.episode} || Reward = {total_reward:.2f} || steps:{scene.steps} || eps = {agent.epsilon:.2f} || {collected}/{scene.reward_batch_size}")
    scene.episode += 1
    # print(f"Episode {scene.episode}: Total Reward = {total_reward:.2f}. Steps:{scene.steps}, Epsilon = {agent.epsilon:.2f}, Loss AVG: {(np.sum(np.array(losses))/len(losses)):.2f}")
    agent.decay_epsilon()
    if collected == scene.num_of_rewards and scene.steps < lowest_steps:
        lowest_steps = scene.steps
        agent.update_target_network()
        print('updating target q function')

_rewards = np.array(rewards)
_steps = np.array(steps).astype(float)
_epsilons = np.array(epsilons)

_rewards /= np.linalg.norm(_rewards)
_steps /= np.linalg.norm(_steps)
_epsilons /= np.linalg.norm(_epsilons)

# Example data
x = list(range(len(steps)))

# Plot the data
plt.figure(figsize=(8, 6))  # Set the figure size
# plt.plot(x, _rewards, label='rewards', color='blue', linestyle='-')
plt.plot(x, _steps, label='steps', color='green', linestyle='-')
# plt.plot(x, _epsilons, label='epsilons', color='green', linestyle=':')

# Add labels, title, and legend
plt.xlabel('Episodes')
plt.legend()

# Save the plot as an image
plt.savefig(f'plot_image_{scene.rnd_value}_{grid_shape}.png', dpi=300)  # Save with high resolution
plt.close()  # Close the figure to avoid displaying it when run


# scene.run()
