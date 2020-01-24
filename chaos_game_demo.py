from chaos_game import *
import sys
from scipy.constants import constants
phi = (1 + 5 ** 0.5) / 2

cg = ChaosGameRegularPolygon(4)
cg.add_virtual_vertex(1)
cg.chaos_game(100000, 2/3)
cg.generate_heatmap(save='sample/Sierpinski carpet.png')