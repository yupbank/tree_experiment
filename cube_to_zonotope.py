import numpy as np
from scipy.spatial import ConvexHull
import itertools


def generator_to_zonotope(generators):
    n, d = generators.shape
    x = np.array(list(itertools.product([0, 1], repeat=n)))
    return x.dot(generators)


def get_zonotopes(all_points):
    hull = ConvexHull(all_points)
    return hull.vertices
