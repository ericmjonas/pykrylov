import numpy as np
import scipy.linalg 
import scipy.sparse.linalg

from util import * 

def arnoldi_with_reortho(A, x_init, J):
    U = np.zeros((len(x_init), J+1))
    U[:, 0] = x_init / norm(x_init)
    h = np.zeros((J+1, J))

    for j in range(J):
        u_j = U[:, j]
        U_j = U[:, :j+1]

        u_next = A.compute_Ax(u_j)
        assert len(u_next.flatten()) == len(x_init)

        h[:j+1, j] = np.dot(U_j.T, u_next)
        # orthogonalize
        u_next = u_next - np.dot(U_j, h[:j+1, j])
        assert len(u_next.flatten()) == len(x_init)

        delta = np.dot(U_j.T, u_next)
        # reorthogonalize

        u_next = u_next - np.dot(U_j, delta)
        assert len(u_next.flatten()) == len(x_init)

        h[:j+1, j] = h[:j+1, j] + delta
        h[j+1, j] = norm(u_next)
        if h[j+1, j] < np.finfo(float).eps:
            return U, h[:J, :J]
        u_next = u_next / h[j+1, j]
        U[:, j+1] = u_next

    return U, h[:J, :J]
