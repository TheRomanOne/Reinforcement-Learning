import pygame
import numpy as np
from utils import color_maps, OBJECTS

# class Character:
#     def __init__(self, position, world_map, cell_size, char_type):
#         # Initialize player at a random position
#         self.world_map = world_map
#         self.cell_size = cell_size
#         self.color = color_maps[char_type]
#         self.char_type = char_type
#         self.x, self.y = position[0], position[1]
#         self.world_map.map[self.x, self.y] = char_type
#         self.walkable = [OBJECTS.land.value]

#     def draw(self, screen):
#         # Draw the player as a red circle
#         pygame.draw.circle(screen, self.color, (self.x * self.cell_size + self.cell_size // 2, self.y * self.cell_size + self.cell_size // 2), self.cell_size // 2)
    
#     def move(self, direction):
#         self.world_map.map[self.x, self.y] = 0
#         valid_move = False
#         if direction == 'up' and self.y > 0 and self.world_map.map[self.x, self.y - 1] in self.walkable:
#             self.y -= 1
#             valid_move = True
#         elif direction == 'down' and self.y < self.world_map.map.shape[1] - 1 and self.world_map.map[self.x, self.y + 1] in self.walkable:
#             self.y += 1
#             valid_move = True
#         elif direction == 'left' and self.x > 0 and self.world_map.map[self.x - 1, self.y] in self.walkable:
#             self.x -= 1
#             valid_move = True
#         elif direction == 'right' and self.x < self.world_map.map.shape[0] - 1 and self.world_map.map[self.x + 1, self.y] in self.walkable:
#             self.x += 1
#             valid_move = True
        
#         overriding = self.world_map.map[self.x, self.y]
#         self.world_map.map[self.x, self.y] = self.char_type

#         return valid_move, overriding

class Entity:
    def __init__(self, position, world_map, cell_size, image_path, char_type=None):
        # Initialize player at a random position
        self.cell_size = cell_size
        self.x, self.y = position[0], position[1]
        self.world_map = world_map
        if char_type is not None:
            self.char_type = char_type
            self.world_map.map[self.x, self.y] = char_type
        self.walkable = [OBJECTS.land.value]

        # Load and scale the character image
        self.image = pygame.image.load(image_path).convert_alpha()  # Use convert_alpha for transparency
        self.image = pygame.transform.scale(self.image, (self.cell_size, self.cell_size))

    def draw(self, screen):
        # Draw the character as an image
        screen.blit(self.image, (self.x * self.cell_size, self.y * self.cell_size))

class Character(Entity):
    def __init__(self, position, world_map, cell_size, img_path, char_type):
        super().__init__(position, world_map, cell_size, img_path, char_type)
        
    def move(self, direction):
        self.world_map.map[self.x, self.y] = 0
        valid_move = False
        if direction == 'up' and self.y > 0 and self.world_map.map[self.x, self.y - 1] in self.walkable:
            self.y -= 1
            valid_move = True
        elif direction == 'down' and self.y < self.world_map.map.shape[1] - 1 and self.world_map.map[self.x, self.y + 1] in self.walkable:
            self.y += 1
            valid_move = True
        elif direction == 'left' and self.x > 0 and self.world_map.map[self.x - 1, self.y] in self.walkable:
            self.x -= 1
            valid_move = True
        elif direction == 'right' and self.x < self.world_map.map.shape[0] - 1 and self.world_map.map[self.x + 1, self.y] in self.walkable:
            self.x += 1
            valid_move = True

        overriding = self.world_map.map[self.x, self.y]
        self.world_map.map[self.x, self.y] = self.char_type

        return valid_move, overriding
    
class Player(Character):
    def __init__(self, position, world_map, cell_size, img_path='assets/robot.png'):
        super().__init__(position, world_map, cell_size, img_path, OBJECTS.player.value)
        self.walkable.append(OBJECTS.reward.value)

class Collectable(Entity):
    def __init__(self, position, cell_size, world_map, char_type, img_path='assets/gold_coins.png'):
        super().__init__(position, world_map, cell_size, img_path, char_type)

class Door(Entity):
    def __init__(self, position, cell_size, open=False, img_path='assets/opening.png'):
        if not open:
            img_path = 'assets/door_closed.png'
        super().__init__(position, None, cell_size, img_path)