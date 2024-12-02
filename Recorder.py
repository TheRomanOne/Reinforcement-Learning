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
    

    def save_media(self, name, title, steps, epsilons, update_indices):
    
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
        create_video(np.array(video), f'{base_name}.mp4', fps=30)
        plot_progress_with_map(f'{base_name}.png', title, steps, epsilons, update_indices, self.screenshot)

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
