from chaosgame.chaos_game import *

phi = (1 + 5 ** 0.5) / 2

cg = ChaosGameRegularPolygon(4)
# cg.add_virtual_vertex(1)
cg.chaos_game_restricted(10000, 0.5, 2)
# cg.chaos_game(100000, 2/3)
cg.generate_heatmap()