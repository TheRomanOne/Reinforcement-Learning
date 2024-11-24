import random, os, warnings
import numpy as np
from NeuralAgent import DQLAgent
from Scene import Scene

warnings.filterwarnings("ignore")
os.system('clear')



cell_size = 40
grid_shape = (15, 10)
window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)
    
scene = Scene(window_shape, grid_shape)
agent = DQLAgent(state_size=scene.get_state().shape, epsilon_decay=0.9965, action_size=4)

episodes = 500
lowest_steps = 99999999
for episode in range(episodes):
    state = scene.reset()
    total_reward = 0
    losses = []
    while len(scene.rewards) > 0:
        action = agent.act(state)

        next_state, reward, done = scene.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train()
        if loss > -1:
            losses.append(loss)
        state = next_state
        total_reward += reward

    scene.episode += 1
    print(f"Episode {scene.episode}: Total Reward = {total_reward:.2f}. Steps:{scene.steps}, Epsilon = {agent.epsilon:.2f}, Loss AVG: {np.array(losses).mean():.2f}")
    # print(f"Episode {scene.episode}: Total Reward = {total_reward:.2f}. Steps:{scene.steps}, Epsilon = {agent.epsilon:.2f}, Loss AVG: {(np.sum(np.array(losses))/len(losses)):.2f}")
    agent.decay_epsilon()
    if scene.steps < lowest_steps:
        lowest_steps = scene.steps
        agent.update_target_network()
        print('updating target q function')

# scene.run()
