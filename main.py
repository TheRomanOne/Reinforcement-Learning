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
    if last_update > 30 or last_update > 10 and steps < lowest_steps*1.1 and agent.epsilon > .1:
        lowest_steps = steps
        to_update = True

    if to_update:
        last_update = 0
        agent.update_target_network()
        update_indices.append(episode)
        print('updating target q function')
    
    return last_update

def run_session(session_name, num_of_rewards, seed, grid_shape):
    print(seed, shape, num_rewards)

    cell_size = 40
    
    # if grid_shape is None:
        # grid_shape = np.random.uniform(7, 20, (2,)).astype(int)
    # grid_shape = (9, 9)
        # grid_shape = (12, 7)# , 12)
        # grid_shape = (7, 12)# , 12)
    # grid_shape = (9, 9)
        # grid_shape = (10, 21)
        # grid_shape = (9, 18)
        # grid_shape = (10, 21)
    # num_of_rewards = 1


    MAX_STEPS = min(1500, 10 * grid_shape[0] * grid_shape[1])
    window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)

    use_convolution=False
    print('Network type:', 'Convolution' if use_convolution else "Fully connected")
    scene = Scene(
        window_shape=window_shape, 
        grid_shape=grid_shape, 
        matrix_state=use_convolution, 
        seed=seed,
        num_of_rewards=num_of_rewards,
        free_space_prob=.55 + .15 * np.random.rand()
    )

    agent = DQLAgent(
        state_size=scene.get_state().shape,
        action_size=4,
        num_of_rewards=num_of_rewards,
        epsilon_decay=0.997,
        learning_rate=0.001,
        gamma=.98,
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
    
    while agent.epsilon > 0.07:
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
            action, entropy = agent.act_with_entropy(state)
            next_state, reward, done = scene.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Render
            scene.draw()
            recorder.record(done)            

            loss = agent.train()
            entropies.append(entropy)
            if loss > -1:
                losses.append(loss)
            state = next_state

            scene.draw_gui({
                'Seed': scene.rnd_value,
                'Title:': session_name,
                'Episode': scene.episode,
                # 'Confidence': f"{(1 - agent.epsilon):.2f}",
                'Entropy': np.round(entropy, 2),
                # 'Reward': np.round(scene.player.reward, 2)
            },
            f_size=25,
            update=True)


        mean_entropy = np.array(entropies).mean()
        agent.adjust_epsilon(mean_entropy)
                
        if done: last_complete = 0
        else: last_complete += 1

        steps.append(scene.steps)
        epsilons.append(agent.epsilon)

        last_update = update_if_required(last_update, lowest_steps, scene.steps, scene.episode, agent, update_indices)

        collected = scene.reward_batch_size - len(scene.rewards)
        print(f"{scene.episode} || steps: {scene.steps} || entropy: {mean_entropy:.2f} || eps: {agent.epsilon:.2f} || {collected}/{scene.reward_batch_size} {" - Pass" if collected==scene.reward_batch_size else ""}{" (REC)" if recorder.captured else ''}")
    
    
    title = f"Seed {scene.rnd_value} || Map {grid_shape} || Rewards {scene.num_of_rewards} || Freespace {scene.free_space_prob:.2f}"
    recorder.save_media(session_name, title, steps, epsilons, update_indices)
    return True



if __name__ == '__main__':
    session_name = "test run"
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