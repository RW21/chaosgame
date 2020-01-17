from chaos_game import *
import sys
from scipy.constants import constants
phi = (1 + 5 ** 0.5) / 2

# cg = ChaosGameRegularPolyhedra(4)
# cg.chaos_game_restricted(10000, 0.5, 0)
# cg.generate_3d_scatter()

cg = ChaosGameRegularPolygon(4)
# cg.add_virtual_vertex(1)
cg.chaos_game(100000, 2/3)
cg.generate_scatter(show_vertexes=True, show_initial_point=True)
# cg.generate_heatmap()