import pygame
import numpy as np
from utils import color_maps, OBJECTS







"""
TODO:
make a class for player inherit from character
make interact function. if interact with reward, find form self.rewards by index

"""












class Character:
    def __init__(self, position, world_map, cell_size, char_type):
        # Initialize player at a random position
        self.world_map = world_map
        self.cell_size = cell_size
        self.color = color_maps[char_type]
        self.char_type = char_type
        self.x, self.y = position[0], position[1]
        self.world_map.map[self.x, self.y] = char_type
        self.walkable = [OBJECTS.land.value]

    def draw(self, screen):
        # Draw the player as a red circle
        pygame.draw.circle(screen, self.color, (self.x * self.cell_size + self.cell_size // 2, self.y * self.cell_size + self.cell_size // 2), self.cell_size // 2)
    
    def move(self, direction):
        self.world_map.map[self.x, self.y] = 0
        
        if direction == 'up' and self.y > 0 and self.world_map.map[self.x, self.y - 1] in self.walkable:
            self.y -= 1
        elif direction == 'down' and self.y < self.world_map.map.shape[1] - 1 and self.world_map.map[self.x, self.y + 1] in self.walkable:
            self.y += 1
        elif direction == 'left' and self.x > 0 and self.world_map.map[self.x - 1, self.y] in self.walkable:
            self.x -= 1
        elif direction == 'right' and self.x < self.world_map.map.shape[0] - 1 and self.world_map.map[self.x + 1, self.y] in self.walkable:
            self.x += 1
        
        overriding = self.world_map.map[self.x, self.y]
        self.world_map.map[self.x, self.y] = self.char_type
        return overriding

        

class Player(Character):
    def __init__(self, position, world_map, cell_size, char_type):
        super().__init__(position, world_map, cell_size, char_type)
        self.reward = 0
        self.walkable.append(OBJECTS.reward.value)

    # def move(self, direction):
    #     overriding = super().move(direction)
    #     if overriding == 4:
    #         self.reward += 1
    #     return overriding

