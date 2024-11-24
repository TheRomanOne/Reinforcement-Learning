import numpy as np
from enum import Enum


_colors = [
    [0.2, .8, 0.3],  # 0 land
    [.75, .37, .17],  # 1 obstable
    [0.3, 0.2, .8],  # 2 player
    [.73, .85, .2],  # 4 reward
    [.8, 0.2, 0.3],  # 3 npc
]
color_maps = [tuple(np.round(np.array(c) * 255).astype(int)) for c in _colors]

class OBJECTS(Enum):
    land = 0
    obstable = 1
    player = 2
    reward = 3
    npc = 4
