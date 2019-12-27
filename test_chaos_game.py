from unittest import TestCase
from chaos_game import generate_fixed_3d_coordinates


class Test(TestCase):
    def test_generate_fixed_3d_coordinates(self):
        assert generate_fixed_3d_coordinates((1, True), (2, False), (3, False)) == [(1, 2, 3), (-1, 2, 3), (1, 2, 3),
                                                                                    (-1, 2, 3)]
