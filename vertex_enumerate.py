import numpy as np
import scipy.optimize as op


def flip(current_vertex, sign_vector, generators, child_flip_index):
    k = child_flip_index
    current_vertex -= sign_vector[k] * generators[k]
    sign_vector[k] *= -1
    return current_vertex, sign_vector


def is_adjacent(sign_vector, generators, child_flip_index):
    k = child_flip_index
    A_eq = np.vstack([generators[:k]*sign_vector[:k, np.newaxis],
                      generators[k+1:]*sign_vector[k+1:, np.newaxis]]).T
    b_eq = generators[k]*sign_vector[k, np.newaxis]
    c = np.ones(A_eq.shape[1])
    res = op.linprog(c, A_eq=A_eq, b_eq=b_eq)
    return not res.success


def dfs(current_vertex, sign_vector, generators, n):
    visited_cells = set()
    sign_flip_stack = [(-1, -1)]
    vertices = []

    while sign_flip_stack:
        current_flip_index, child_flip_index = sign_flip_stack.pop()
        while child_flip_index < n-1:
            child_flip_index += 1
            if is_adjacent(sign_vector, generators, child_flip_index):
                current_vertex, sign_vector = flip(
                    current_vertex, sign_vector, generators, child_flip_index)
                if tuple(sign_vector) not in visited_cells:
                    vertices.append([i for i in current_vertex])
                    visited_cells.add(tuple(sign_vector))
                    sign_flip_stack.append((child_flip_index, -1))
                    break
                current_vertex, sign_vector = flip(
                    current_vertex, sign_vector, generators, child_flip_index)
        if child_flip_index == n:
            if sign_flip_stack:
                sign_flip_stack.pop()
            if current_flip_index != -1:
                current_vertex, sign_vector = flip(
                    current_vertex, sign_vector, generators, current_flip_index)
    return np.array(vertices)


def main():
    d, n = 4, 5
    np.random.seed = 2
    generators = np.random.uniform(size=(n, d))
    random_point = np.ones(d)
    sign_vector = np.sign(generators.dot(random_point))
    current_vertex = (sign_vector > 0).T.dot(generators)
    res = dfs(current_vertex, sign_vector, generators, n)
    print(len(res))
    print(res)
    #one_side = np.cumsum(sorted(generators, key=lambda x: -x[0]/x[1]), axis=0)
    #the_other_side = np.cumsum(
    #    sorted(generators, key=lambda x: x[0]/x[1]), axis=0)
    #print(one_side)
    #print(the_other_side)


if __name__ == "__main__":
    main()
