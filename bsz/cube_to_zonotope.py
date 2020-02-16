import numpy as np


def enumerate_all_points(generators):
    import itertools
    n, d = generators.shape
    x = np.array(list(itertools.product([0, 1], repeat=n)))
    return x.dot(generators)


def vertex_points(all_points):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(all_points)
    return all_points[hull.vertices]


def num_vertices(m, n):
    from scipy.special import comb
    return 2 * sum([comb(m-1, i) for i in range(n)])


# def unique_rows(S):
#     T = S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
#     return np.unique(T).view(S.dtype).reshape(-1, S.shape[1])


def bool_sign(a):
    return np.where(np.sign(a) > 0, 1.0, 0.0)


def sample_zonotope_vertices(generators, n_samples=50, n_batches=50):
    m, n = generators.shape
    res = np.empty([0, m])
    for i in range(n_batches):
        samples = np.random.normal(size=(n_samples, n))
        S = np.sign(samples.dot(generators.T))
        res = np.unique(np.vstack((res, S, -S)), axis=0)
    S = bool_sign(res)
    return S.dot(generators), S

    #num_verts = num_vertices(m, n)
    #if num_verts > S.shape[0]:
    #    print('Warning: {} of {} vertices found.'.format(
    #        S.shape[0], num_verts))


def aggregate_generators_by_direction(x):
    normal_x = x/np.linalg.norm(x, axis=1, keepdims=True)
    unique_vector, index, inverse = np.unique(
        normal_x, axis=0, return_index=True, return_inverse=True)
    res = np.zeros_like(unique_vector)
    np.add.at(res, inverse, x)
    return res, inverse
