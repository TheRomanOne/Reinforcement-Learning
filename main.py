import os
from NeuralAgent import DQLAgent
from Scene import Scene
from utils import device, capture_screenshot_as_array, plot_progress_with_map, create_video
import pygame
import warnings
import numpy as np
from datetime import datetime
import argparse

def check_q():
    q = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            q = event.key == pygame.K_q
            break
    return q

def run_session(num_of_rewards, seed, grid_shape):
    print(seed, shape, num_rewards)

    cell_size = 40
    
    # if grid_shape is None:
        # grid_shape = np.random.uniform(7, 20, (2,)).astype(int)
    # grid_shape = (10, 10)
        # grid_shape = (12, 7)# , 12)
        # grid_shape = (7, 12)# , 12)
        # grid_shape = (7, 7)
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
        epsilon_decay=0.99,
        learning_rate=0.001,
        gamma=.98,
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
    n = 95
    recording_checkpoints = [1.1] + [(n-i)/n for i in range(n)]
    recording = False
    rec_counter = 0
    max_frames_in_cut = 100
    videos = []

    video_lens = []
    while agent.epsilon > 0.05:
        random_player_position=False#.93 < agent.epsilon < .98

        use_seed = True#agent.epsilon < .5 or agent.epsilon > .95
        valid_scene = False
        while not valid_scene:
            try:
                state = scene.reset(use_seed, random_player_position)
                valid_scene = True
            except:
                print("Invalid scene. searching for a new one")

        if screenshot is None:
            screenshot = capture_screenshot_as_array(scene.screen)


        total_reward = 0
        losses = []
        scene.episode += 1

        current_session_frames = [screenshot]
        captured = False
        while scene.steps < MAX_STEPS and len(scene.rewards) > 0:
            if check_q():
                return False                 

            action = agent.act(state)
            next_state, reward, done = scene.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            scene.draw()

            if scene.steps == 1 and not captured and len(recording_checkpoints) > 0 and agent.epsilon < recording_checkpoints[0]:
                recording = True
            
            if recording:
                current_session_frames.append(capture_screenshot_as_array(scene.screen))
                rec_counter = len(current_session_frames)
                if done and rec_counter <= max_frames_in_cut:
                    if rec_counter not in video_lens:
                        video_lens.append(rec_counter)
                        captured = True
                        recording = False
                        if scene.steps > max_frames_in_cut:
                            print('')
                        # add a few frames of finished state
                        for _ in range(4):
                            current_session_frames.append(current_session_frames[-1])

                        videos.append(current_session_frames)
                    recording_checkpoints = recording_checkpoints[1:]
                    current_session_frames = []
                elif len(current_session_frames) == max_frames_in_cut:
                    recording = False
                    current_session_frames = []
            
            scene.draw_gui({
                'Episode': scene.episode,
                'Confidence': f"{(1 - agent.epsilon):.2f}",
            },
            update=True)

            loss = agent.train()
            if loss > -1:
                losses.append(loss)
            state = next_state
            total_reward += reward

        current_session_frames = []
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

        print(f"{scene.episode} || steps: {scene.steps} || eps: {agent.epsilon:.2f} || {collected}/{scene.reward_batch_size} {" - Pass" if collected==scene.reward_batch_size else ""}{" (REC)" if captured else ''}")

        if last_update > 3 or (len(dones) > 7 and sum(dones[:-7]) > 4):
            agent.decay_epsilon()

        to_update=False
        if last_update > 30 or last_update > 10 and scene.steps < lowest_steps*1.1 and agent.epsilon > .1:
            lowest_steps = scene.steps
            to_update = True

        if to_update:
            last_update = 0
            agent.update_target_network()
            update_indices.append(scene.episode)
            print('updating target q function')


    title = f"Seed {scene.rnd_value} || Map {grid_shape} || Rewards {scene.num_of_rewards} || Freespace {scene.free_space_prob:.2f}"
    
    video = []
    videos.sort(key=len)
    print('Total runs in video:', len(videos))
    print('Frames per scene', [len(v) for v in videos])
    max_len = 900
    for v in videos:
        if len(video) < max_len:
            video = v + video 

    # video = [item for sublist in videos for item in sublist]
    if update_indices[-1] == len(epsilons):
        update_indices = update_indices[:-1]
        steps = steps[:-1]
    save_media(video, screenshot, title, steps, epsilons, update_indices)
    return True

def save_media(frames, screenshot, title, steps, epsilons, update_indices):
    base_name = img_name = f"plots/{datetime.now().strftime("%d%m%y_%H%M%S")}"
    create_video(np.array(frames), f'{base_name}.mp4', 20)
    plot_progress_with_map(f'{base_name}.png', title, steps, epsilons, update_indices, screenshot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="<seed> ")

    # Make arguments optional with default values
    parser.add_argument('--s', type=int, nargs='?', default=None, help="seed number")
    parser.add_argument('--w', type=int, nargs='?', default=-1, help="width")
    parser.add_argument('--h', type=int, nargs='?', default=-1, help="height")
    parser.add_argument('--n', type=int, nargs='?', default=1, help="number of rewards")

    # Parse the arguments
    args = parser.parse_args()

    width = args.w
    height = args.h
    if width == -1 or height == -1:
        shape = np.random.uniform(7, 20, (2,)).astype(int)
    else:
        shape = (width, height)

    # Access the arguments
    num_rewards = max(1, args.n)
    seed = args.s
    go_on = True
    while go_on:
        warnings.filterwarnings("ignore")
        os.system('clear')
        print("Device:", device)
        if run_session(num_rewards, seed, shape):
            go_on = False