import numpy as np
import scipy.linalg 
import scipy.sparse.linalg

from util import * 

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
