import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import constants
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point


def get_eq_triangle_height(side: int):
    return (np.sqrt(3 * side ** 2)) / 2


def generate_initial_triangle():
    return np.array([0, 1, np.sqrt(3) / 2])


def random_point_inside_triangle(triangle):
    r1 = np.random.uniform(0, 2)
    r2 = np.random.uniform(0, 2)

    return [np.sqrt(r1) * (1 - r2), r2 * np.sqrt(r1)]


def chaos_game_triangle(iteration, factor, absolute=False):
    triangle = np.array([[0, 0], [80, 0], [40, 20]])

    points_x = []
    points_y = []

    position = random_point_inside_triangle(triangle)

    for i in range(iteration):
        if absolute:
            position = np.absolute((triangle[np.random.randint(0, 3)] - position)) * factor
        else:
            position = ((triangle[np.random.randint(0, 3)] - position)) * factor

        points_x.append(position[0])
        points_y.append(position[1])

    plt.scatter(points_x, points_y, s=0.5)

    plt.show()


def random_point_inside_square(square):
    square_side = square[1][0] - square[0][0]
    return [np.random.uniform(0, square_side), np.random.uniform(0, square_side)]


def chaos_game_square(iteration, factor, absolute=False):
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    points_x = []
    points_y = []

    position = random_point_inside_square(square)

    for i in range(iteration):
        current_vertex = np.random.randint(0, 3)
        if absolute:
            position = np.absolute(square[current_vertex] - position) * factor
        else:
            position = (square[current_vertex] - position) * factor

        points_x.append(position[0])
        points_y.append(position[1])

    plt.scatter(points_x, points_y, s=0.5)
    # plt.savefig('sample_4.png', dpi=1000)

    plt.show()


def generate_random_point_in_polygon(polygon):
    poly = Polygon(polygon)
    min_x, min_y, max_x, max_y = poly.bounds

    while True:
        random_point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if random_point.within(poly):
            return random_point


class ChaosGame3d:
    def __init__(self):
        self.x, self.y, self.z = [], [], []

    def chaos_game(self, iteration, factor, absolute=False):
        position = np.array([0, 0, 0])

        for i in range(iteration):
            current_vertex = self.vertexes[np.random.randint(len(self.vertexes))]
            if absolute:
                position = np.absolute(current_vertex - position) * factor
            else:
                position = (current_vertex - position) * factor

            self.x.append(position[0])
            self.y.append(position[1])
            self.z.append(position[2])

        # todo fix so np array from start
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

    def generate_3d_scatter(self, fig, show=True, size=0.5):
        ax = Axes3D(fig)
        ax.scatter(self.x, self.y, self.z, s=0.5)

        if show:
            plt.show()

        return fig

    def generate_cross_section(self, exclude: str) -> (np.array, np.array):
        """
        Generates a cross section, excluding the axis (x,y,z) in the parameter.
        :param exclude:
        :return:
        """
        if exclude == 'x':
            return self.y, self.z

        if exclude == 'y':
            return self.x, self.z

        if exclude == 'z':
            return self.x, self.y


def generate_fixed_3d_coordinates(a, b, c):
    """
    All inputs are in format of (value, is_fixed).
    :param a:
    :param b:
    :param c:
    """
    count = 0

    if a[1]:
        count += 1
    if b[1]:
        count += 1
    if c[1]:
        count += 1

    num_combinations = 2 ** count

    def generate(parameter) -> list:
        coordinate = []
        multiplier = -1
        for i in range(num_combinations):
            if parameter[1]:
                coordinate = [parameter[0]] * num_combinations
            else:
                multiplier *= -1
                coordinate.append(multiplier * parameter[0])

        return coordinate

    coordinates_x = generate(a)
    coordinates_y = generate(b)
    coordinates_z = generate(c)

    return [(coordinates_x[i], coordinates_y[i], coordinates_z[i]) for i in range(len(coordinates_x))]


class ChaosGameRegularPolyhedra(ChaosGame3d):
    def __init__(self, faces):
        super().__init__()
        if faces not in {4, 8, 6, 20, 12}:
            raise RegularPolyhedronNotPossible
        else:
            self.faces = faces
        self.vertexes = []
        self.generate_vertexes()

    def generate_vertexes(self):
        # https://en.wikipedia.org/wiki/Platonic_solid#Cartesian_coordinates

        # not clean todo use dict
        if self.faces == 4:
            self.vertexes = [(1, 1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1)]

        if self.faces == 8:
            self.vertexes = generate_fixed_3d_coordinates((1, False), (0, True), (0, True)) \
                            + generate_fixed_3d_coordinates((0, True), (1, False), (0, True)) \
                            + generate_fixed_3d_coordinates((0, True), (0, True), (1, True))

        if self.faces == 6:
            self.vertexes = tuple(itertools.product([1, -1], repeat=3))

        if self.faces == 12:
            self.vertexes = tuple(itertools.product([1, -1], repeat=3)) \
                            + generate_fixed_3d_coordinates((0, True), (1 / constants.golden, False),
                                                            (constants.golden, False)) \
                            + generate_fixed_3d_coordinates((1 / constants.golden, False), (constants.golden, False),
                                                            (0, True)) \
                            + generate_fixed_3d_coordinates((constants.golden, False),
                                                            (0, True), (1 / constants.golden, False))

        if self.faces == 20:
            self.vertexes = generate_fixed_3d_coordinates((0, True), (1, False), (constants.golden, False)) \
                            + generate_fixed_3d_coordinates((1, False), (constants.golden, False), (0, True)) \
                            + generate_fixed_3d_coordinates((constants.golden, False), (0, True), (1, False))


class ChaosGame2dBase:
    def __init__(self):
        self.x = []
        self.y = []

    def generate_heatmap(self, show=True, save=False, colormap: cm = cm.jet, sigma=2) -> plt:

        if len(self.x) == 0 or len(self.y) == 0:
            raise PointsNotGenerated('Points are not generated')

        img, extent = myplot(self.x, self.y, 2)
        plt.imshow(img, extent=extent, origin='lower', cmap=colormap)

        if save:
            plt.savefig('sample_5.png', dpi=500)

        if show:
            plt.show()

        return plt

    def generate_scatter(self, show=True, save=False, size=0.05) -> plt:
        if len(self.x) == 0 or len(self.y) == 0:
            raise PointsNotGenerated('Points are not generated.')

        plt.scatter(self.x, self.y, s=size)

        if save:
            plt.savefig('out.png', dpi=500)

        if show:
            plt.show()

        return plt


class ChaosGame2d(ChaosGame2dBase):
    def __init__(self, polygon):
        super().__init__()
        self.polygon = generate_polygon(polygon)

    def chaos_game(self, iteration, factor, absolute=False):
        position = generate_random_point_in_polygon(self.polygon)

        for i in range(iteration):
            current_vertex = np.random.randint(0, len(self.polygon))

            if absolute:
                position = (self.polygon[current_vertex] - position) * factor
            else:
                position = (self.polygon[current_vertex] - position) * factor

            self.x.append(position[0])
            self.y.append(position[1])

    def get_random_vertex(self):
        return self.polygon[np.random.randint(len(self.polygon))]

    def chaos_game_restricted(self, iteration, factor, restriction, absolute=False):
        position = generate_random_point_in_polygon(self.polygon)
        previous = np.zeros(2)
        new = self.polygon[np.random.randint(len(self.polygon))]

        if restriction == 'currently chosen cannot be chosen':
            for i in range(iteration):
                while (new == previous).all():
                    new = self.polygon[np.random.randint(len(self.polygon))]

                previous = new

                if absolute:
                    position = np.absolute(new - position) * factor
                else:
                    position = (new - position) * factor

                self.x.append(position[0])
                self.y.append(position[1])

        elif restriction == 'vertex cannot be two places away':

            for i in range(iteration):
                new_vertex = self.get_random_vertex()

                while new_vertex in get_vertexes_apart_from(self.polygon, new_vertex)[2]:
                    new_vertex = self.get_random_vertex()

                current_vertex = new_vertex

                if absolute:
                    position = np.absolute(current_vertex - position) * factor
                else:
                    position = (current_vertex - position) * factor

                self.x.append(position[0])
                self.y.append(position[1])


def get_vertexes_apart_from(vertexes: np.array, vertex) -> dict:
    distances = {}

    # assumes no duplicates in vertexes
    current_index = np.where(vertexes == vertex)[0][0]

    for i, vertex in enumerate(vertexes):
        distances[np.absolute(current_index - i)] = vertex

    return distances


def generate_polygon(vertex):
    N = vertex
    r = 10000000
    x = []
    y = []

    for n in range(0, vertex):
        x.append(r * np.cos(2 * np.pi * n / N))
        y.append(r * np.sin(2 * np.pi * n / N))

    return np.array([[x[i], y[i]] for i in range(len(x))])


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def subplot_for_polyhedra_cross_section(cg: ChaosGameRegularPolyhedra):
    fig = plt.figure()

    size = 10
    fig.set_figheight(size)
    fig.set_figwidth(size)

    plt.subplot(4, 1, 1)
    cg.generate_3d_scatter(fig)

    scatter_size = 0.5

    x, y = cg.generate_cross_section('x')

    cg_base = ChaosGame2dBase()
    cg_base.x = x
    cg_base.y = y

    plt.subplot(4, 1, 2)
    cg_base.generate_scatter(show=False, size=scatter_size)

    x, y = cg.generate_cross_section('y')

    cg_base = ChaosGame2dBase()
    cg_base.x = x
    cg_base.y = y

    plt.subplot(4, 1, 3)
    cg_base.generate_scatter(show=False, size=scatter_size)

    x, y = cg.generate_cross_section('z')

    cg_base = ChaosGame2dBase()
    cg_base.x = x
    cg_base.y = y

    plt.subplot(4, 1, 4)
    cg_base.generate_scatter(show=False, size=scatter_size)

    fig.show()


class ChaosGameMultidimensionBase:
    def __init__(self, dimension: int):
        coordinates = [[] for i in range(dimension)]


class PointsNotGenerated(Exception):
    def __init__(self, message):
        super().__init__(message)


class RegularPolyhedronNotPossible(Exception):
    def __init__(self):
        super().__init__(
            '''A regular polyhedron cannot be generated with the number of faces.
             A regular polyherdon can only have 4, 6, 8, 12, 20 faces.''')

# demo_1 = ChaosGameRegularPolyhedra(8)
#
# demo_1.chaos_game(10000, -1/constants.golden)
# demo_1.generate_3d_scatter()
