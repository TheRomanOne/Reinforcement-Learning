import os
from NeuralAgent import DQLAgent
from Scene import Scene
from utils import device
import pygame
import warnings
import numpy as np
from datetime import datetime
import argparse
from Recorder import Recorder

def check_q():
    q = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            q = event.key == pygame.K_q
            break
    return q

def update_if_required(last_update, lowest_steps, steps, episode, agent, update_indices) :
    last_update += 1
    to_update=False
    if last_update > 20 or last_update > 5 and steps < lowest_steps*1.1:
        lowest_steps = steps
        to_update = True

    if to_update:
        last_update = 0
        agent.update_target_network()
        update_indices.append(episode)
        print('updating target q function')
    
    return last_update, lowest_steps

def run_session(session_name, num_of_rewards, seed, grid_shape, cell_size=40):
    print('Session name:', session_name)
    print(seed, shape, num_rewards)
    # if grid_shape is None:
        # grid_shape = np.random.uniform(7, 20, (2,)).astype(int)
    # grid_shape = (9, 9)
        # grid_shape = (12, 7)# , 12)
        # grid_shape = (7, 12)# , 12)
    # grid_shape = (9, 9)
        # grid_shape = (10, 21)
        # grid_shape = (9, 18)
    # grid_shape = (9, 12)
    # num_of_rewards = 3


    MAX_STEPS = min(1000, 10 * grid_shape[0] * grid_shape[1])
    window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)

    use_convolution=False
    print('Network type:', 'Convolution' if use_convolution else "Fully connected")
    scene = Scene(
        window_shape=window_shape, 
        grid_shape=grid_shape, 
        matrix_state=use_convolution, 
        seed=seed,
        num_of_rewards=num_of_rewards,
        free_space_prob=.65 + .15 * np.random.rand()
    )

    agent = DQLAgent(
        state_size=scene.get_state().shape,
        action_size=4,
        num_of_rewards=num_of_rewards,
        epsilon_decay=0.9975,
        learning_rate=0.001,
        gamma=.95,
        use_convolution=use_convolution,
        epsilon_min=.05,
        epsilon_max = .9
    )

    steps = []
    epsilons = []
    update_indices = []

    last_complete = 0
    last_update = 0
    lowest_steps = 999
    use_seed = False

    recorder = Recorder(agent)
    recorder.assign_screen(scene.screen)
    # all_entropies = []

    continue_session = True

    target_entropy = 0
    target_entropies = []
    entropy_means = []
    dones = []
    rewards = []
    while continue_session:
        random_player_position=False#.93 < agent.epsilon < .98

        use_seed = True#agent.epsilon < .5 or agent.epsilon > .95
        valid_scene = False
        while not valid_scene:
            try:
                state = scene.reset(use_seed, random_player_position)
                valid_scene = True
            except:
                print("Invalid scene. searching for a new one")


        losses = []
        scene.episode += 1

        recorder.start_new_session()
        entropies = []
        while scene.steps < MAX_STEPS and len(scene.rewards) > 0:
            if check_q():
                return False                 

            # action = agent.act(state)
            action, entropy = agent.act(state)
            entropies.append(entropy)
            target_entropy += (entropy - target_entropy) / 2
            next_state, reward, done, collected_reward = scene.step(action)
            dones.append(done)

            if not collected_reward:
                agent.store_transition(state, action, reward, next_state, done)
            else:
                agent.reduce_memory()
                scene.state_hashes = set()

            # Render
            scene.draw()
            recorder.record(done)            

            scene.draw_gui({
                'Title:': session_name,
                'Seed': scene.rnd_value,
                'Episode': scene.episode,
                'Epsilon': f"{(agent.epsilon):.2f}",
                'Entropy': np.round(entropy, 2),
                'Reward': np.round(scene.player.reward, 2)
            },
            f_size=25,
            update=True)

            loss = agent.train()
            if loss > -1:
                losses.append(loss)
            state = next_state

        # all_entropies.append(entropies)
        mean_entropy = np.array(entropies).mean()
        entropy_means.append(mean_entropy)
                
        if done: last_complete = 0
        else: last_complete += 1

        steps.append(scene.steps)
        epsilons.append(agent.epsilon)

        if agent.epsilon > .15:
            last_update, lowest_steps = update_if_required(last_update, lowest_steps, scene.steps, scene.episode, agent, update_indices)

        collected = scene.reward_batch_size - len(scene.rewards)
        continue_session = agent.epsilon > 0.07
        # if target_entropy == -1:
        #     target_entropy = mean_entropy
        # target_entropy += (mean_entropy.item() - target_entropy) / 2
        
        target_entropies.append(target_entropy)
        rewards.append(scene.player.reward)
        # agent.adjust_epsilon(mean_entropy, target_entropy)
        if len(rewards) > 10 and last_update > 3 or (len(dones) > 7 and sum(dones[:-7]) > 4):
            panelty = (lowest_steps - np.array(steps[-5:]).mean())/10
            agent.adjust_epsilon(entropy)
            rewards.pop(0)
            if len(dones) > 10: dones.pop(0)

        print(f"{scene.episode} || steps: {scene.steps} || entropy: {mean_entropy:.2f} || reward: {scene.player.reward:.2f} || eps: {agent.epsilon:.2f} || {collected}/{scene.reward_batch_size} {" - Pass" if collected==scene.reward_batch_size else ""}{" (REC)" if recorder.captured else ''}")
    
    
    # title = f"Seed {scene.rnd_value} || Map {grid_shape} || Rewards {scene.num_of_rewards} || Freespace {scene.free_space_prob:.2f}"
    target_entropies = np.array(target_entropies)
    target_entropies -= target_entropies.min()
    target_entropies /= target_entropies.max()

    entropy_means = np.array(entropy_means)
    entropy_means -= entropy_means.min()
    entropy_means /= entropy_means.max()
    recorder.save_media(session_name, steps, epsilons, target_entropies, entropy_means, update_indices)
    return True



if __name__ == '__main__':
    session_name = "test run 2"
    parser = argparse.ArgumentParser(description="<seed> ")
    # Make arguments optional with default values
    parser.add_argument('--s', type=int, nargs='?', default=None, help="seed number")
    parser.add_argument('--w', type=int, nargs='?', default=-1, help="width")
    parser.add_argument('--h', type=int, nargs='?', default=-1, help="height")
    parser.add_argument('--r', type=int, nargs='?', default=1, help="number of rewards")
    parser.add_argument('--n', type=str, nargs='?', default=session_name, help="session name")

    # Parse the arguments
    args = parser.parse_args()
    session_name = args.n
    width = args.w
    height = args.h
    if width == -1 or height == -1:
        shape = np.random.uniform(7, 20, (2,)).astype(int)
    else:
        shape = (width, height)

    # Access the arguments
    num_rewards = max(1, args.r)
    seed = args.s
    go_on = True
    while go_on:
        warnings.filterwarnings("ignore")
        os.system('clear')
        print("Device:", device)
        if run_session(session_name, num_rewards, seed, shape):
            go_on = False