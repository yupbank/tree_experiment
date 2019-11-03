from collections import defaultdict
from scipy.special import comb
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from scipy.spatial import ConvexHull
import itertools


def generator_to_zonotope(generators):
    n, d = generators.shape
    x = np.array(list(itertools.product([0, 1], repeat=n)))
    return x.dot(generators)


def get_zonotopes(all_points):
    hull = ConvexHull(all_points)
    return all_points[hull.vertices]


def num_vertices(m, n):
    return 2 * sum([comb(m-1, i) for i in range(n)])


def unique_rows(S):
    T = S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
    return np.unique(T).view(S.dtype).reshape(-1, S.shape[1])


def bool_sign(a): return np.where(np.sign(a) > 0, 1.0, 0.0)


def zonotope_vertices(generators, n_samples=50, n_batches=50):
    m, n = generators.shape
    res = np.empty([0, m])
    for i in range(n_batches):
        samples = np.random.normal(size=(n_samples, n))
        S = np.sign(samples.dot(generators.T))
        res = unique_rows(np.vstack((res, S, -S)))
    S = bool_sign(res)
    return S.dot(generators)

    #num_verts = num_vertices(m, n)
    #if num_verts > S.shape[0]:
    #    print('Warning: {} of {} vertices found.'.format(
    #        S.shape[0], num_verts))


def deduplicate(x):
  normal_x = x/np.linalg.norm(x, axis=1)[:, np.newaxis]
  sx, sy = np.nonzero(np.isclose(normal_x.dot(normal_x.T), 1.0))
  visited = set()
  collect = defaultdict(set)
  for i, j in zip(sx, sy):
    if j in visited:
      continue
    else:
      visited.add(j)
    collect[i].add(j)
  return np.vstack([np.sum(x[list(k)], axis=0) for k in collect.values()])
