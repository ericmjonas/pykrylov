import numpy as np
import util
import irlba
import lanczos


def test_simple():
    np.random.seed(0)
    N = 30
    M = 20
    A = np.random.normal(0, 1, (N, M))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)

    x_init = np.random.normal(0, 1, M)
    x_init = x_init / util.norm(x_init)
    m = 10
    k = 7
    ### FIRST WE DO RAW LANCZOS
    U, s, V = np.linalg.svd(A)
    P, Q, B, rm  = lanczos.partial_lanczos_bidiagonalization(opA, x_init, m)
    U_l, s_l, V_l = np.linalg.svd(B)
    l_percent_diff = util.min_dist_in_frac(s_l, s)


    # then IRLBA
    my_U, my_s, my_V = irlba.svd(opA, x_init, k, m, 1e-6)
    
    percent_diff = util.min_dist_in_frac(my_s, s)
    print l_percent_diff
    print percent_diff
