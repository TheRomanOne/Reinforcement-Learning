import pygame
import numpy as np
from utils import device, plot_progress_with_map, create_video

class Recorder:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.max_frames_in_cut = []
        self.videos = []
        self.video_lens = []
        self.recording = False
        self.rec_counter = 0
        self.max_frames_in_cut = 100
        self.video_lens = []
        n = 95
        self.recording_checkpoints = [1.1]+[(n-i)/n for i in range(n)]
    

    def save_media(self, name, title, steps, epsilons, entropies, update_indices):
    
        s = 5
        videos = self.videos[:s] + self.videos[s:][::4] + self.videos[-3:]
        video = []
        videos.sort(key=len)
        print('Total runs in video:', len(videos))
        max_len = 900
        for v in videos:
            if len(video) < max_len:
                video = v + video 

        update_indices = update_indices[:-1]
        base_name =  f"plots/{name.replace(' ', '_')}"
        # create_video(np.array(video), f'{base_name}.mp4', fps=30)
        

        episodes = np.array(range(len(entropies)))
        steps = np.array(steps)
        plot = [(episodes, steps, 'green', 'Steps')]
        scatter = [(episodes[update_indices], steps[update_indices], 10, 'red', 'Updated Indices')]

        plot_progress_with_map(
            f'{base_name}_steps.png', title,
            map_screenshot=self.screenshot,
            x_label='Episode',
            y_label='Steps"',
            to_scatter=scatter,
            to_plot=plot,
        )
        # plot_progress_with_map(, title, "Act", 'Entropy', 'blue', entropies, range(len(entropies)), update_indices, self.screenshot)

        start_session = []
        end_session = []
        ents = []
        means = []
        for e in entropies:
            start_session.append(len(ents))
            end_session.append(len(ents) + len(e) - 1)
            ents = ents + e
            means.append(np.array(e).mean())

        ents = np.array(ents)
        
        # plot entropies
        acts = np.array(range(len(ents)))
        # plot = [(acts, ents, 'blue', 'Steps')]
        # scatter = [
        #     (acts[start_session], ents[start_session], 20, 'red', 'Level Begin'),
        #     (acts[end_session], ents[end_session], 20, 'green', 'Level End')
        # ]
        
        # plot entropy means
        plot = [(episodes, means, 'blue', 'Entropy mean')]
        scatter = [(episodes[update_indices], plot[update_indices], 10, 'red', 'Updated Indices')]

        plot_progress_with_map(
            f'{base_name}_entropy.png',
            title='Entropy',
            map_screenshot=self.screenshot,
            x_label='Act',
            y_label='Entropy"',
            to_scatter=scatter,
            to_plot=plot,
        )

    def start_new_session(self):
        self.new_session = True
        self.captured = False
        self.recording = False
        self.screenshot = self.get_screenshot()
        self.current_session_frames = [self.screenshot]

    def assign_screen(self, screen):
        self.screen = screen

    def get_screenshot(self):
        """Capture the pygame screen as a NumPy array."""
        screenshot = pygame.surfarray.array3d(self.screen)
        screenshot = np.transpose(screenshot, (1, 0, 2))  # Transpose to (height, width, 3)
        return screenshot

    def _check_condition(self):
        first_step = self.new_session and not self.captured
        check_point = len(self.recording_checkpoints) > 0 and self.agent.epsilon < self.recording_checkpoints[0]
        return first_step and check_point

    def record(self, done):
        if not self.recording:
            self.recording = self._check_condition()
        
        if self.recording:
            self.current_session_frames.append(self.get_screenshot())
            rec_counter = len(self.current_session_frames)
            if done and rec_counter <= self.max_frames_in_cut:
                if rec_counter not in self.video_lens:
                    self.video_lens.append(rec_counter)
                    self.captured = True
                    self.recording = False

                    # add a few frames of finished state
                    for _ in range(4):
                        self.current_session_frames.append(self.current_session_frames[-1])

                    self.videos.append(self.current_session_frames)
                self.recording_checkpoints = self.recording_checkpoints[1:]
                self.current_session_frames = []
            elif len(self.current_session_frames) == self.max_frames_in_cut:
                self.recording = False
                self.current_session_frames = []
        
        self.new_session = False
