import os
from NeuralAgent import DQLAgent
from Scene import Scene
from utils import device, capture_screenshot_as_array, plot_progress_with_map, create_video
import pygame
import warnings
import numpy as np
from datetime import datetime

def check_q():
    q = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            q = event.key == pygame.K_q
            break
    return q

def run_session():
    cell_size = 40
    
    # grid_shape = np.random.uniform(5, 20, (2,)).astype(int)
    # grid_shape = (5, 5)
    grid_shape = (14, 7)# , 12)
    # grid_shape = (7, 7)
    # grid_shape = (12, 12)
    # grid_shape = (20, 20)

    MAX_STEPS = min(1500, 10 * grid_shape[0] * grid_shape[1])
    window_shape = (grid_shape[0] * cell_size, grid_shape[1] * cell_size)

    use_convolution=True
    print('Network type:', 'Convolution' if use_convolution else "Fully connected")
    scene = Scene(
        window_shape=window_shape, 
        grid_shape=grid_shape, 
        matrix_state=use_convolution, 
        # seed=2671,
        # seed=868,
        num_of_rewards=2,
        free_space_prob=.55 + .15 * np.random.rand()
    )

    agent = DQLAgent(
        state_size=scene.get_state().shape,
        action_size=4,
        epsilon_decay=0.99,
        learning_rate=0.001,
        gamma=.95,
        use_convolution=use_convolution
    )

    rewards = []
    steps = []
    epsilons = []
    update_indices = [0]
    screenshot = None
    last_complete = 0
    last_update = 0
    dones = []
    lowest_steps = 999
    use_seed = False

    # Prepare video handling
    n = 10
    recording_checkpoints = [(n-i)/n for i in range(n)] + [.05, .02]
    recording = False
    rec_counter = 0
    max_frames_in_cut = 70
    video = []

    while agent.epsilon > 0.01:
        random_player_position=False#.93 < agent.epsilon < .98

        use_seed = True#agent.epsilon < .5 or agent.epsilon > .95
        
        state = scene.reset(use_seed, random_player_position)
        if screenshot is None:
            screenshot = capture_screenshot_as_array(scene.screen)


        total_reward = 0
        losses = []
        scene.episode += 1

        while scene.steps < MAX_STEPS and len(scene.rewards) > 0:
            if check_q():
                return False                 

            action = agent.act(state)

            next_state, reward, done = scene.step(action)
            scene.draw(gui={
                'Episode': scene.episode,
                'Confidence': f"{(1 - agent.epsilon):.2f}",
            })
            if not recording and len(recording_checkpoints) > 0 and agent.epsilon < recording_checkpoints[0]:
                recording = True
                rec_counter = 0
            
            if recording:
                video.append(capture_screenshot_as_array(scene.screen))
                rec_counter += 1
            
            if rec_counter == max_frames_in_cut:
                recording = False
                rec_counter = 0
                recording_checkpoints = recording_checkpoints[1:]

            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train()
            if loss > -1:
                losses.append(loss)
            state = next_state
            total_reward += reward

        
        if done:
            last_complete = 0
        else:
            last_complete += 1
        last_update += 1
        dones.append(done)
        rewards.append(total_reward)
        steps.append(scene.steps)
        epsilons.append(agent.epsilon)

        collected = scene.reward_batch_size - len(scene.rewards)
        print(f"{scene.episode} || steps: {scene.steps} || eps: {agent.epsilon:.2f} || {collected}/{scene.reward_batch_size} {" - Pass" if collected==scene.reward_batch_size else ""}{"(REC)" if recording else ''}")

        if last_update > 3 or (len(dones) > 7 and sum(dones[:-7]) > 4):
            agent.decay_epsilon()
        # success = sum(dones[-30:])  / len(dones[-30:])

        to_update=False
        # if agent.epsilon > .6:
        #     if last_update > 30:
        #         to_update = True
        if last_update > 30 or last_update > 10 and scene.steps < lowest_steps*1.1 and agent.epsilon > .1:
            lowest_steps = scene.steps
            to_update = True

        if to_update:
            # lowest_steps = scene.steps
            # last_complete = 0
            last_update = 0
            agent.update_target_network()
            update_indices.append(scene.episode)
            print('updating target q function')


    title = f"Seed {scene.rnd_value} || Map {grid_shape} || Rewards {scene.num_of_rewards} || Freespace {scene.free_space_prob:.2f}"
    save_media(video, screenshot, title, steps, epsilons, update_indices)
    return True

def save_media(frames, screenshot, title, steps, epsilons, update_indices):
    base_name = img_name = f"plots/{datetime.now().strftime("%d%m%y_%H%M%S")}"
    create_video(np.array(frames), f'{base_name}.mp4', 24)
    plot_progress_with_map(f'{base_name}.png', title, steps, epsilons, update_indices, screenshot)

while True:
    warnings.filterwarnings("ignore")
    os.system('clear')
    print("Device:", device)
    if run_session():
        break
    