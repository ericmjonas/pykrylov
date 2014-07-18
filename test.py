import numpy as np
import scipy.linalg 
import scipy.sparse.linalg

from util import * 

def arnoldi_with_reortho(A, x_init, J):
    U = np.zeros((len(x_init), J+1))
    U[:, 0] = x_init / norm(x_init)
    h = np.zeros((J+1, J))

    for j in range(J):
        print "j=", j
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
        print "||u_next|| = ", norm(u_next)
        if h[j+1, j] < np.finfo(float).eps:
            return U, h[:J, :J]
        u_next = u_next / h[j+1, j]
        print u_next.shape, U[:, j+1].shape
        U[:, j+1] = u_next

    return U, h[:J, :J]

if __name__ == "__main__":

    np.random.seed(0)

    N = 10000
    A = np.random.normal(0, 1, (N, N))
    A = np.dot(A.T, A)
    e_val, e_vect =  scipy.sparse.linalg.eigs(A)

    Aop = LinOp(A)

    J = 1000
    x_init = np.random.normal(0, 1, N)
    x_init = x_init / norm(x_init)
    U, h = arnoldi_with_reortho(Aop, x_init, J)
    h_e_val, h_e_vect = scipy.linalg.eig(h)
    print h
    close = 1e-6
    close_num = 0
    for ev in e_val:
        if np.min(np.abs(h_e_val - ev)) < close:
            close_num += 1
    print "there are", close_num, "close eigenvalues" 
    print np.max(h_e_val), np.max(e_val)
