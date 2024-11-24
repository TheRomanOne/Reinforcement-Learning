import pygame
from NeuralAgent import DQLAgent
from Scene import Scene

# grid_shape = (20, 15)
cell_size = 50
grid_shape = (12, 10)
window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)
    
scene = Scene(window_shape, grid_shape)
agent = DQLAgent(state_size=scene.get_state().shape[0], epsilon_decay=0.995, action_size=4)

episodes = 500
for episode in range(episodes):
    state = scene.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = scene.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        total_reward += reward

    scene.episode += 1
    print(f"Episode {scene.episode}: Total Reward = {total_reward}. Epsilon = {agent.epsilon:.2f}")
    agent.decay_epsilon()
    agent.update_target_network()  # Update target network periodically

# scene.run()
