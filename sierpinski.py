import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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


class ChaosGame:
    def __init__(self, polygon):
        self.polygon = generate_polygon(polygon)

    def chaos_game(self, iteration, factor, absolute=False):
        points_x = []
        points_y = []

        position = random_point_inside_square(self.polygon)

        for i in range(iteration):
            current_vertex = np.random.randint(0, len(self.polygon))

            if absolute:
                position = np.absolute(self.polygon[current_vertex] - position) * factor
            else:
                position = (self.polygon[current_vertex] - position) * factor

            points_x.append(position[0])
            points_y.append(position[1])


        # plt.scatter(points_x, points_y, s=0.05)

        img, extent = myplot(points_x, points_y, 8)
        plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        # plt.set_title("Smoothing with  $\sigma$ = %d" % s)
        plt.savefig('sample_5.png', dpi=500)

        plt.show()


        # plt.show()


def generate_polygon(vertex):
    N = vertex
    r = 100000
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

ChaosGame(10).chaos_game(10000000, -0.5, absolute=False)
# chaos_game_triangle(20000, 0.5)
# chaos_game_square(20000, 0.5, absolute=True)
