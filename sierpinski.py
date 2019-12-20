import numpy as np
import matplotlib.pyplot as plt


def get_eq_triangle_height(side: int):
    return (np.sqrt(3 * side ** 2)) / 2


def generate_initial_triangle():
    return np.array([0, 1, np.sqrt(3) / 2])


def sierpinski(n):
    initial = generate_initial_triangle()
    one = np.array([initial, [initial[1], initial[1] + 1, get_eq_triangle_height(initial[1] - initial[0])], ])
    print(one)
    # for i in range(n):


def create_random_point_inside_triangle(triangle):
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)

    return [np.sqrt(r1) * (1 - r2), r2 * np.sqrt(r1)]


def chaos_game(n):
    triangle = np.array([[0, 0], [1, 0], [0, 1]])

    position = create_random_point_inside_triangle(triangle)

    points_x = []
    points_y = []

    for i in range(n):
        position = np.absolute((triangle[np.random.randint(0, 3)] - position)) / 2
        points_x.append(position[0])
        points_y.append(position[1])


    plt.scatter(points_x, points_y)
    plt.show()





chaos_game(100000)
