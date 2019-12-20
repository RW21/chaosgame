import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def get_eq_triangle_height(side: int):
    return (np.sqrt(3 * side ** 2)) / 2


def generate_initial_triangle():
    return np.array([0, 1, np.sqrt(3) / 2])


def sierpinski(n):
    initial = generate_initial_triangle()
    one = np.array([initial, [initial[1], initial[1] + 1, get_eq_triangle_height(initial[1] - initial[0])], ])
    print(one)


def random_point_inside_triangle(triangle):
    r1 = np.random.uniform(0, 2)
    r2 = np.random.uniform(0, 2)

    return [np.sqrt(r1) * (1 - r2), r2 * np.sqrt(r1)]


def chaos_game(n):
    triangle = np.array([[0, 0], [1000, 0], [0, 1000]])

    points_x = []
    points_y = []
    points = []

    position = random_point_inside_triangle(triangle)

    for i in range(n):
        position = ((triangle[np.random.randint(0, 3)] - position)) / 2
        points_x.append(position[0])
        points_y.append(position[1])
        # points.append(position)

    # plt.figure(dpi=1200)
    # plt.scatter(points_x, points_y, s=0.5)
    plt.scatter(points_x, points_y, s=0.5)

    plt.savefig('sample_1.png', dpi=1200)
    plt.show()


chaos_game(20000)
