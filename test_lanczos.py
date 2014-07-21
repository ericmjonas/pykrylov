import numpy as np

import lanczos
import util

np.set_printoptions(precision=3)


def test_k_lanczos_bidiag():
    N = 1000
    A = np.random.normal(0, 1, (N, N))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)
    x_init = np.random.normal(0, 1, N)

    B, U, V = lanczos.lanczos_bidiagonalization(opA, x_init, 100)
    print B.shape

    U_l, s_l, V_l = np.linalg.svd(B)
    s_l_big_to_small = np.sort(s_l)[::-1]
    percent_diff = util.min_dist_in_frac(s_l_big_to_small, s)*100
    print percent_diff
    # check for < 1% error
    np.testing.assert_array_less(np.sort(percent_diff)[:10], np.ones(10))


def test_m_partial_lanczos_bidiag():
    N = 10
    A = np.random.normal(0, 1, (N, N))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)
    x_init = np.random.normal(0, 1, N)
    x_init = x_init / util.norm(x_init)
    m = 3

    P, Q, B, rm  = lanczos.partial_lanczos_bidiagonalization(opA, x_init, m)
    
    print B
    U_l, s_l, V_l = np.linalg.svd(B.T)

    percent_diff = util.min_dist_in_frac(s_l, s)
    print percent_diff

    print np.dot(P[:, m-1].T, rm) 
    
    
    
