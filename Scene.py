import pygame
import numpy as np
import random
from utils import color_maps, OBJECTS
from Character import Player, Collectable, Door

class Map:
    def __init__(self, width, height, map_size, rnd_value, free_space_prob):
        # Create the grid with obstacles (1 for obstacles, 0 for free space)
        self.width = width
        self.height = height
        self.map_size = map_size
        self.map = np.random.choice([0, 1], size=(width, height), p=[free_space_prob, 1-free_space_prob])
        self.rnd_value = rnd_value
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
                basec_olor = np.array(basec_olor) * (.97 + .01 * (np.sin(x * y * self.rnd_value)))

                reward_color = np.array(color_maps[OBJECTS.reward.value])
                if _map[x, y] == 0:
                    a = 1 - self.reward_map[x, y]

                    color = (basec_olor + .645 * (a * np.zeros((3,)) + (1 - a) * reward_color).astype(int)).tolist()
                    color = np.clip(color, 0, 255)
                else:
                    color = basec_olor

                pygame.draw.rect(screen, color, (x * self.map_size, y * self.map_size, self.map_size, self.map_size))



class Scene:
    def __init__(self, window_shape, grid_shape, matrix_state=False, seed=None, num_of_rewards=1, free_space_prob=.9):
        # Initialize Pygame and game window
        pygame.init()
        self.grid_shape = grid_shape
        self.window_shape = window_shape
        self.screen = pygame.display.set_mode((window_shape[0], window_shape[1]))
        pygame.display.set_caption("Window")
        self.local_rng = np.random.default_rng()  # Independent random generator
        self.clock = pygame.time.Clock()
        self.episode = 0
        self.matrix_state = matrix_state
        self.font = pygame.font.Font(None, 36)  # Use default font with size 36
        self.free_space_prob = free_space_prob
        # Create game components
        self.cell_size = min(window_shape[0]/grid_shape[0], window_shape[1]/grid_shape[1])
        self.rnd_value = np.random.randint(9999) if seed is None else seed
        print(f"Seed: {self.rnd_value}")
        self.num_of_rewards = num_of_rewards
        self.state = self.reset()


    def reset(self, use_seed=True, random_player_position=False):
        self.steps = 0
        if use_seed:
            np.random.seed(self.rnd_value)
            random.seed(self.rnd_value)
        self.map = Map(self.grid_shape[0], self.grid_shape[1], self.cell_size, self.rnd_value, self.free_space_prob)
        self.init_npcs(0)
        self.init_rewards(self.num_of_rewards, batches=random_player_position)
        pos = self.get_random_position(true_random=random_player_position)
        self.player = Player(position=pos, world_map=self.map, cell_size=self.cell_size)
        self.open_door = Door((self.player.x, self.player.y), self.cell_size, open=True)
        self.draw()#add_gui=False)
        return self.get_state()
    
    def get_state(self):
        state = self.map.map.flatten()

        size = 3 # land / obstacle, player, reward
        _state = np.zeros((state.shape[0], size))
        state = state - 1
        _state[np.arange(state.shape[0]), state] = 1
        
        if self.matrix_state:
            state = _state.astype(int).reshape(self.grid_shape[0], self.grid_shape[1], 3)
        else:
            state = _state.flatten()

        return state
          
    def get_random_position(self, true_random=False):
        free_cells = np.array(self.map.get_free_cells())
        if true_random:
            index = self.local_rng.choice(len(free_cells), size=1, replace=False)[0]
        else:
            index = np.random.choice(range(free_cells.__len__()), size=1, replace=False)[0]
        return free_cells[index]

    def init_npcs(self, n):
        npcs = []
        # for _ in range(n):
        #     pos = self.get_random_position()
        #     npcs.append(Character(pos, self.map, self.cell_size, char_type=OBJECTS.npc.value))
        self.npcs =  npcs

    def get_free_neighbors(self, pos):
        r = [-1, 0, 1]
        neighbors = []
        for i in r:
            for j in r:
                if abs(i) + abs(j) < 2:
                    x = np.clip(pos[0] + i, 0, self.grid_shape[0]-1)
                    y = np.clip(pos[1] + j, 0, self.grid_shape[1]-1)
                    if  self.map.map[x, y] != OBJECTS.obstable.value:
                        neighbors.append((x, y))
        
        return neighbors


    def get_reward_effect_area(self, pos):

        effect = np.zeros_like(self.map.reward_map)
        dist = 7
        visited = []
        neighbors = self.get_free_neighbors(pos)
        while len(neighbors) > 0:
            n = neighbors[0]
            x, y = n[0], n[1]
            neighbors = neighbors[1:]

            if n not in visited:
                visited.append(n)
                value = min(dist, np.linalg.norm(np.array([x, y]) - pos))
                value = dist - value
                value = effect[x, y] + .5 * value / dist
                effect[x, y] = value

                neighbors = neighbors + self.get_free_neighbors(n)
                neighbors = list(set(neighbors))
        return effect
        
    def get_reward_effect_area_old(self, pos):
        effect = np.zeros_like(self.map.reward_map)
        dist = 7

        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                value = min(dist, np.linalg.norm(np.array([x, y]) - pos))
                value = dist - value
                value = effect[x, y] + .5 * value / dist
                effect[x, y] = value
        return effect

    def init_rewards(self, n, batches):
        rewards = []
        positions = []
        for _ in range(n):
            pos = self.get_random_position()
            positions.append(pos)
        
        if batches:
            batch_s = 1 + self.local_rng.integers(0, n, size=1)[0]
            indices = self.local_rng.choice(range(len(positions)), size=batch_s, replace=False)
            positions = [positions[i] for i in indices]
            self.reward_batch_size = batch_s
        else:
            self.reward_batch_size = n

        for pos in positions:
            rewards.append(Collectable(pos, self.cell_size, world_map=self.map, char_type=OBJECTS.reward.value))
        positions = np.array(positions).T
        
        for pos in positions.T:
            effect = self.get_reward_effect_area(pos)
            self.map.reward_map += effect

        self.rewards = rewards
    
    def step(self, action):
        self.steps += 1

        if action == 0:
            valid_move, overriding = self.player.move('up')
        elif action == 1:
            valid_move, overriding = self.player.move('down')
        elif action == 2:
            valid_move, overriding = self.player.move('left')
        elif action == 3:
            valid_move, overriding = self.player.move('right')
        
        if not valid_move:
            reward = -1
        elif overriding == OBJECTS.reward.value:
            # Remove reward effect area 
            self.rewards = [rw for rw in self.rewards if not (rw.x == self.player.x and rw.y == self.player.y)]
            self.map.reward_map -= self.get_reward_effect_area(np.array([self.player.x, self.player.y]))
            
            reward = .5#(self.num_of_rewards - len(self.rewards))/self.num_of_rewards
        else:
            reward = -0.1

        # End condition
        done = False
        if len(self.rewards) == 0:
            done = True
            reward = 1
        next_state = self.get_state()

        return next_state, reward, done

    def draw(self, gui=None):
        self.screen.fill(color_maps[OBJECTS.land.value])
        
        # Draw the grid and player
        self.map.draw(self.screen)

        self.open_door.draw(self.screen)

        self.player.draw(self.screen)
        for rw in self.rewards:
            rw.draw(self.screen)

        for npc in self.npcs:
            npc.draw(self.screen)

        if gui is not None:
            self.draw_gui(gui)
        
        # Update the display
        pygame.display.update()

    def draw_gui(self, msg):
        # Create a smaller font for the GUI
        f_size = 20
        small_font = pygame.font.Font(None, f_size)  # Adjust the size as needed (smaller number = smaller font)
        
        # Define the color for the text (e.g., light blue)
        text_color = (50, 50, 50)  # RGB for light blue
        
        # Format the GUI message
        for i, (name, value) in enumerate(msg.items()):
            gui_text = ''.join([f"{name}: {value}" ])
        
            # Render the text surface with the specified font and color
            text_surface = small_font.render(gui_text, True, text_color)
        
            # Position the text at the top-left corner
            text_position = (10, 10 + f_size * i)
            self.screen.blit(text_surface, text_position)    
