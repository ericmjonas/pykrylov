import numpy as np
import util
import irlba
import lanczos


def test_simple():
    np.random.seed(0)
    N = 300
    M = 200
    A = np.random.normal(0, 1, (N, M))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)

    x_init = np.random.normal(0, 1, M)
    x_init = x_init / util.norm(x_init)

    k = 7
    m = 5*k

    ### FIRST WE DO RAW LANCZOS
    U, s, V = np.linalg.svd(A)
    # what are the true largest singular values
    s_true_max = s[np.argsort(s)[-k:]]

    P, Q, B, rm  = lanczos.partial_lanczos_bidiagonalization(opA, x_init, m)

    U_l, s_l, V_l = np.linalg.svd(B)
    s_l = np.sort(s_l)[::-1]
    s_l = s_l[:k]
    l_percent_diff = util.min_dist_in_frac(s_l, s_true_max)


    # then IRLBA
    my_U, my_s, my_V = irlba.svd(opA, x_init, k, m, 1e-6)
    
    my_s = np.sort(my_s)[::-1]
    percent_diff = util.min_dist_in_frac(my_s, s_true_max)
    print "The lanczos results are" 
    print l_percent_diff

    print "The irlba results are"
    print percent_diff
