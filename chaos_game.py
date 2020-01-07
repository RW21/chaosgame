import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import constants
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point


# todo refactor


def generate_random_point_in_polygon(polygon):
    """Generates a randomly uniform point inside a polygon.

    Args:
        polygon (np.array): Numpy array with all vertexes.

    Returns:
        A shapely point inside the polygon.

    """
    poly = Polygon(polygon)
    min_x, min_y, max_x, max_y = poly.bounds

    while True:
        random_point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if random_point.within(poly):
            return random_point


def add_points(*args):
    """Add points to the object.

    Args:
        *args: Coordinate list followed by the point to add.

    Examples:
        >>> add_points(self.x, self.y, [1,2])
    """
    add = args[-1]

    for i, arg in enumerate(args[:-1]):
        arg.append(add[i])


class ChaosGame3d:
    """Chaos game in 3d

    Attributes:
        x, y, z: Numpy arrays of all the coordinates.

    """

    def __init__(self):
        self.x, self.y, self.z = [], [], []

    def generate_3d_scatter(self, fig, show=True, size=0.5):
        """Generates a 3d scatter plot from coordinates.

        Args:
            fig:
            show:
            size:

        Returns:

        """
        ax = Axes3D(fig)
        ax.scatter(self.x, self.y, self.z, s=size)

        if show:
            plt.show()

        return fig

    def generate_cross_section(self, exclude: str) -> (np.array, np.array):
        """Generates a cross section, excluding the axis (x,y,z) in the parameter.

        Args:
            exclude (str): Which axis to exclude.

        Returns:

        """
        if exclude == 'x':
            return self.y, self.z

        if exclude == 'y':
            return self.x, self.z

        if exclude == 'z':
            return self.x, self.y


def generate_fixed_3d_coordinates(a, b, c):
    """Creates fixed 3d coordinates

    All inputs are in format of (value, is_fixed).

    Args:
        a:
        b:
        c:

    Returns (list): Coordinates.

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
    """Chaos game in a regular polyhedra

    Attributes:
        faces: Number of faces of the polyhedra.
        vertexes: Coordinates of the polyhedra.
    """

    def __init__(self, faces):
        super().__init__()
        if faces not in {4, 8, 6, 20, 12}:
            raise RegularPolyhedronNotPossible
        else:
            self.faces = faces
        self.vertexes = []
        self.generate_vertexes()

    def get_random_vertex(self):
        """Get random vertex from current vertexes.

        Returns: A random vertex.

        """
        return self.vertexes[np.random.randint(len(self.vertexes))]

    def generate_vertexes(self):
        """Creates vertex arrays of the polyhedra.

        """
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

    def chaos_game(self, iteration, factor, absolute=False):
        """Generates chaos game coordinates.

        Args:
            iteration: Number of iterations.
            factor:
            absolute: If absolute should be used.
        """
        # instead of choosing a random point in polyhedron, set initial point to be origin
        position = np.zeros(3)

        for i in range(iteration):
            current_vertex = self.vertexes[np.random.randint(len(self.vertexes))]
            if absolute:
                position = np.absolute(current_vertex - position) * factor
            else:
                position = (current_vertex - position) * factor

            self.x.append(position[0])
            self.y.append(position[1])
            self.z.append(position[2])

    def chaos_game_restricted(self, iteration, factor, restriction, absolute=False):
        """A restricted version of the 3d chaos game.

        Restrictions:

        Currently chosen cannot be chosen -> 0


        Args:
            iteration:
            factor:
            restriction:
            absolute:
        """
        if self.x or self.y or self.z:
            raise PointsAlreadyGenerated

        # instead of choosing a random point in polyhedron, set initial point to be origin
        position = np.zeros(3)

        vertex = self.get_random_vertex()
        new_vertex = self.get_random_vertex()

        if restriction == 1:

            for i in range(iteration):
                while vertex == new_vertex:
                    new_vertex = self.get_random_vertex()

                vertex = new_vertex

                if absolute:
                    position = np.absolute(vertex - position) * factor
                else:
                    position = (vertex - position) * factor

                add_points(self.x, self.y, self.z, position)


class ChaosGame2dBase:
    """

    Attributes:
        x: x coordinates.
        y: y coordinates.
        vectors: Vertexes of outline of the polygon.
        polygon: Points of vertexes of the polygon.
    """

    def __init__(self):
        self.x = []
        self.y = []

        self.vectors: np.array = None
        self.polygon: np.array = None

    def generate_heatmap(self, show=True, save='', colormap: cm = cm.jet, sigma=2) -> plt:
        """Generates heatmap graph from the coordinates.

        Args:
            show: To show or not.
            save: To save or not.
            colormap: Type of Matplotlib colormap to use.
            sigma: Sigma to use. Higher sigma creates an intense graph.

        Returns: Generated plot.

        """
        if len(self.x) == 0 or len(self.y) == 0:
            raise PointsNotGenerated('Points are not generated')

        img, extent = myplot(self.x, self.y, 2)
        plt.axis('off')
        plt.imshow(img, extent=extent, origin='lower', cmap=colormap)

        if save:
            plt.savefig(save, dpi=500)

        if show:
            plt.show()

        return plt

    def generate_scatter(self, show=True, save='', size=0.05) -> plt:
        """Generates a scatter plot from coordinates.

        Args:
            show: To show or not.
            save: To save or not.
            size: Size of each point.

        Returns: Generated plot.

        """
        if len(self.x) == 0 or len(self.y) == 0:
            raise PointsNotGenerated('Points are not generated.')

        plt.scatter(self.x, self.y, s=size)
        plt.axis('off')

        if save:
            plt.savefig(save, dpi=500)

        if show:
            plt.show()

        return plt

    def generate_polygon_outline(self):
        """Generates vertexes of the polygon.

        A polygon needs to be generated first.
        """

        if not self.polygon.size:
            raise PointsNotGenerated

        self.vectors = np.zeros(len(self.polygon))

        for i in range(len(self.polygon)):
            self.vectors[i] = (self.polygon[i] - self.polygon[i - 1])

    def outline(self):
        if not self.vectors:
            self.generate_polygon_outline()

        origin = [0], [0]  # origin point

        plt.quiver(*origin, self.vectors[:, 0], self.vectors[:, 1], color=['r', 'b', 'g'], scale=21)
        plt.show()


class ChaosGameRegularPolygon(ChaosGame2dBase):
    """Chaos game with regular polygon base.

    Attributes:
        polygon: Number of vertexes.

    """

    def __init__(self, polygon):
        super().__init__()

        if self.polygon is None:
            self.polygon = generate_polygon(polygon)

    def chaos_game(self, iteration, factor, absolute=False):
        factor *= -1

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
        """Picks and returns a random vertex from the current vertexes.

        Returns:

        """
        return self.polygon[np.random.randint(len(self.polygon))]

    def chaos_game_restricted(self, iteration, factor, restriction, absolute=False):
        """A restricted version of the 2d chaos game.

        Restrictions:

        Currently chosen cannot be chosen -> 0

        Vertex cannot be two places away -> 1

        Current vertex cannot be chosen next -> 2


        Args:
            iteration: Number of iterations.
            factor: Multiplication factor.
            restriction: Restriction number as listed above.
            absolute: If absolute or not.
        """
        factor *= -1
        position = generate_random_point_in_polygon(self.polygon)
        previous = np.zeros(2)
        new = self.polygon[np.random.randint(len(self.polygon))]

        if restriction == 0:
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

        elif restriction == 1:
            for i in range(iteration):
                new_vertex = self.get_random_vertex()

                while (get_vertexes_apart_from(self.polygon, new_vertex)[2][0] == new_vertex).all():
                    new_vertex = self.get_random_vertex()

                current_vertex = new_vertex

                if absolute:
                    position = np.absolute(current_vertex - position) * factor
                else:
                    position = (current_vertex - position) * factor

                self.x.append(position[0])
                self.y.append(position[1])

        elif restriction == 2:
            current_vertex = self.get_random_vertex()

            for i in range(iteration):
                new_vertex = self.get_random_vertex()

                while compare_arrays(new_vertex, current_vertex):
                    new_vertex = self.get_random_vertex()

                current_vertex = new_vertex

                if absolute:
                    position = np.absolute(current_vertex - position) * factor
                else:
                    position = (current_vertex - position) * factor

                current_vertex = new_vertex

                self.x.append(position[0])
                self.y.append(position[1])


def compare_arrays(np1: np.array, np2: np.array) -> bool:
    return (np1 == np2).all()


def get_vertexes_apart_from(vertexes: np.array, vertex) -> dict:
    distances = {}

    # assumes no duplicates in vertexes
    current_index = np.where(vertexes == vertex)[0][0]

    for i, vertex in enumerate(vertexes):
        distance = np.absolute(current_index - i)

        # wanted to use a set :(
        if distance not in distances:
            distances[distance] = []

        distances[distance].append(vertex)

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


class NotGenerated(Exception):
    pass


class PolygonNotGenerated(NotGenerated):
    def __init__(self):
        super().__init__(
            '''
            Polygon is not generated
            '''
        )


class PointsNotGenerated(NotGenerated):
    def __init__(self, message):
        super().__init__(message)


class IteartionNotEnough(Exception):
    def __init__(self):
        super().__init__(
            '''
            At least 10000 iterations are required to produce a clear fractal.
            '''
        )


class RegularPolyhedronNotPossible(Exception):
    def __init__(self):
        super().__init__(
            '''A regular polyhedron cannot be generated with the number of faces.
             A regular polyherdon can only have 4, 6, 8, 12, 20 faces.''')


class PointsAlreadyGenerated(Exception):
    def __init__(self):
        super().__init__(
            '''Points have already been generated.'''
        )
