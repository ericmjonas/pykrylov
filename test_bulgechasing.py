import numpy as np
from nose.tools import * 
import bulgechasing
import util

def test_bc():
    N = 4
    np.set_printoptions(precision=3)
    np.random.seed(0)
    B = np.arange(N*N) + 1
    B = np.random.permutation(B)
    B.shape = N, N

    B = np.triu(B)

    B = np.tril(B, 1)

    print B

    Bnew = bulgechasing.bulgechasing_gk_svd_step(B, 0)
    b_eigs = np.linalg.eig(np.dot(B, B.T))[0]

    Bnew_eigs =  np.linalg.eig(np.dot(Bnew, Bnew.T))[0]
    assert np.max(util.min_dist_in(b_eigs, Bnew_eigs)) < 1e-6
    
