import pygame
import numpy as np
import random
from utils import color_maps, OBJECTS
from Character import Character, Player

class Map:
    def __init__(self, width, height, map_size):
        # Create the grid with obstacles (1 for obstacles, 0 for free space)
        self.width = width
        self.height = height
        self.map_size = map_size
        self.map = np.random.choice([0, 1], size=(width, height), p=[0.93, 0.07])
        land_coords = self.get_free_cells()
        self.collectables = land_coords[np.random.choice(land_coords.shape[0], size=10, replace=False)]
        self.rnd_value = np.random.rand()
        self.reward_map = np.zeros_like(self.map).astype(float)
        
    def get_free_cells(self):
        land_indices = np.where(self.map == 0)
        return np.array(list(zip(land_indices[0], land_indices[1])))

    
    def draw(self, screen):
        # Draw the grid cells
        for x in range(self.width):
            for y in range(self.height):
                _map = np.array(self.map)
                _map[_map > 1] = 0

                basec_olor = color_maps[_map[x, y]]
                basec_olor = np.array(basec_olor) * (.97 + .03 * (np.sin(x * y * self.rnd_value)))

                reward_color = np.array(color_maps[OBJECTS.reward.value])
                if _map[x, y] == 0:
                    a = 1 - self.reward_map[x, y]

                    color = (basec_olor + .565 * (a * np.zeros((3,)) + (1 - a) * reward_color).astype(int)).tolist()
                    color = np.clip(color, 0, 255)
                else:
                    color = basec_olor

                pygame.draw.rect(screen, color, (x * self.map_size, y * self.map_size, self.map_size, self.map_size))



class Scene:
    def __init__(self, window_shape, grid_shape):
        # Initialize Pygame and game window
        pygame.init()
        self.grid_shape = grid_shape
        self.window_shape = window_shape
        self.screen = pygame.display.set_mode((window_shape[0], window_shape[1]))
        pygame.display.set_caption("Window")
        self.clock = pygame.time.Clock()
        self.episode = 1
        # Font for the GUI
        self.font = pygame.font.Font(None, 36)  # Use default font with size 36

        # Create game components
        self.cell_size = min(window_shape[0]/grid_shape[0], window_shape[1]/grid_shape[1])

        self.state = self.reset()

    def reset(self):
        self.map = Map(self.grid_shape[0], self.grid_shape[1], self.cell_size)
        pos = self.get_random_position()
        self.player = Player(position=pos, world_map=self.map, cell_size=self.cell_size, char_type=OBJECTS.player.value)
        self.init_npcs(0)
        self.init_rewards(10)
        return self.get_state()
    
    def get_state(self):
        state = self.map.map.flatten()

        # size = 4 # land, obstacle, player, reward
        # _state = np.zeros((state.shape[0], size))
        # _state[np.arange(state.shape[0]), state] = 1
        # state = _state.astype(int).reshape(self.grid_shape[0], self.grid_shape[1], 4)

        return state
          
    def get_random_position(self):
        free_cells = np.array(self.map.get_free_cells())
        index = np.random.choice(range(free_cells.__len__()), size=1, replace=False)[0]
        cell = free_cells[index]
        return cell

    def init_npcs(self, n):
        npcs = []
        for _ in range(n):
            pos = self.get_random_position()
            npcs.append(Character(pos, self.map, self.cell_size, char_type=OBJECTS.npc.value))
        self.npcs =  npcs


    def init_rewards(self, n):
        rewards = []
        positions = []
        for _ in range(n):
            pos = self.get_random_position()
            rewards.append(Character(pos, self.map, self.cell_size, char_type=OBJECTS.reward.value))
            positions.append(pos)
        positions = np.array(positions).T
        

        dist = 3
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                for pos in positions.T:
                    value = min(dist, np.linalg.norm(np.array([x, y]) - pos))
                    value = dist - value
                    value = self.map.reward_map[x, y] + .5 * value / dist
                    self.map.reward_map[x, y] = value

        self.rewards = rewards
    
    # def handle_input(self):
    #     keys = pygame.key.get_pressed()
    #     overriding = -1
    #     if keys[pygame.K_w]:
    #         overriding = self.player.move('up')
    #     if keys[pygame.K_s]:
    #         overriding = self.player.move('down')
    #     if keys[pygame.K_a]:
    #         overriding = self.player.move('left')
    #     if keys[pygame.K_d]:
    #         overriding = self.player.move('right')
    #     _x, _y = self.player.x, self.player.y
        
    #     reward = 0
    #     if overriding == OBJECTS.reward.value:

    #         for rw in self.rewards:
    #             if rw.x == _x and rw.y == _y:
    #                 reward = self.map.reward_map[rw.x, rw.y] * 100
    #                 if reward > 0 and reward < 100:
    #                     print(reward)
    #                 self.rewards.remove(rw)
    #                 del rw
    #                 break

    #     done = len(self.rewards) == 0
    #     next_state = self.get_state()
    #     return next_state, reward, done
    
    def update(self):
        # Update the game state
        next_state, reward, done = self.handle_input()
        self.player.reward += reward
        for npc in self.npcs:
            if np.random.rand() < .025:
                choice = np.random.choice(['up', 'down', 'left', 'right'], 1)
                npc.move(choice)
        
    def step(self, action):
        # current reward
        reward = self.map.reward_map[self.player.x, self.player.y]
        if action == 0:
            overriding = self.player.move('up')
        elif action == 1:
            overriding = self.player.move('down')
        elif action == 2:
            overriding = self.player.move('left')
        elif action == 3:
            overriding = self.player.move('right')
        
        new_reward = self.map.reward_map[self.player.x, self.player.y]
        reward = -1 + (new_reward - reward) * 100
        if overriding == OBJECTS.reward.value:
            self.rewards = [rw for rw in self.rewards if not (rw.x == self.player.x and rw.y == self.player.y)]
            effect = np.zeros_like(self.map.reward_map)
            pos = np.array([self.player.x, self.player.y])
            dist = 3
            for x in range(self.grid_shape[0]):
                for y in range(self.grid_shape[1]):
                    value = min(dist, np.linalg.norm(np.array([x, y]) - pos))
                    value = dist - value
                    value = effect[x, y] + .5 * value / dist
                    effect[x, y] = value
            self.map.reward_map -= effect

        # End condition
        done = len(self.rewards) == 0
        next_state = self.get_state()
        self.player.reward += reward
        self.draw()
        return next_state, reward, done

    def draw(self):
        self.screen.fill(color_maps[OBJECTS.land.value])
        
        # Draw the grid and player
        self.map.draw(self.screen)


        self.player.draw(self.screen)
        for rw in self.rewards:
            rw.draw(self.screen)

        for npc in self.npcs:
            npc.draw(self.screen)

        # Draw GUI (floating number)
        self.draw_gui()
        
        # Update the display
        pygame.display.update()

    def draw_gui(self):
        # Render the text
        gui_text = f"Episode: {self.episode}, Reward: {self.player.reward:.2f}"
        text_surface = self.font.render(gui_text, True, (255, 255, 255))  # White text
        text_position = (10, 10)  # Top-left corner
        self.screen.blit(text_surface, text_position)
    
    def run(self):
        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update and draw the scene
            self.update()
            self.draw()
            
            # Set FPS (frames per second)
            self.clock.tick(20)
        
        # Quit Pygame
        pygame.quit()
