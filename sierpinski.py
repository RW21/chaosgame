import numpy as np
import matplotlib.pyplot as plt


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
    plt.show()
    plt.savefig('sample_4.png', dpi=700)



# chaos_game_triangle(20000, 0.75, absolute=True)
chaos_game_square(20000, 0.5, absolute=True)
