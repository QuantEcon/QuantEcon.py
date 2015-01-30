import numpy as np
import re

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def gen_rw_mat(n):
    "Generate an n x n matrix of zeros and ones."
    Q = np.random.randn(n, n) - 0.8
    Q = np.where(Q > 0, 1, 0)
    # Make sure that no row contains only zeros
    for i in range(n):
        if Q[i, :].sum() == 0:
            Q[i, np.random.randint(0, n, 1)] = 1
    return Q


def adj_matrix_to_dot(Q, outfile='/tmp/foo_out.dot'):
    """
    Convert an adjacency matrix to a dot file.
    """
    n = Q.shape[0]
    f = open(outfile, 'w')
    f.write('digraph {\n')
    for i in range(n):
        for j in range(n):
            if Q[i, j]:
                f.write('   {0} -> {1};\n'.format(alphabet[i], alphabet[j]))
    f.write('}\n')
    f.close()


def dot_to_adj_matrix(node_num, infile='/tmp/foo_out.dot'):
    Q = np.zeros((node_num, node_num), dtype=int)
    f = open(infile, 'r')
    lines = f.readlines()
    f.close()
    edges = lines[1:-1]  # Drop first and last lines
    for edge in edges:
        from_node, to_node = re.findall('\w', edge)
        i, j = alphabet.index(from_node), alphabet.index(to_node)
        Q[i, j] = 1
    return Q


def adj_matrix_to_markov(Q):
    n = Q.shape[0]
    P = np.empty((n, n))
    for i in range(n):
        P[i, :] = Q[i, :] / float(Q[i, :].sum())
    return P
