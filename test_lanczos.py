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
    
    
    
def test_m_partial_lanczos_bidiag_rect():
    N = 15
    M = 10
    A = np.random.normal(0, 1, (N, M))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)
    x_init = np.random.normal(0, 1, M)
    x_init = x_init / util.norm(x_init)
    m = 3

    P, Q, B, rm  = lanczos.partial_lanczos_bidiagonalization(opA, x_init, m)
    
    print B
    U_l, s_l, V_l = np.linalg.svd(B.T)

    percent_diff = util.min_dist_in_frac(s_l, s)
    print percent_diff

    # sanity check Lanczos identities eqn 1.3-1.5
    assert np.sum(np.abs(opA.compute_Ax(P) - np.dot(Q, B))) < 1e-9
    assert np.sum(np.abs(opA.compute_ATx(Q) - np.dot(P, B.T) - np.outer(rm, np.eye(m)[-1]))) < 1e-9
    print np.dot(P[:, m-1].T, rm) 
    
    
    
def test_first_ablanzbd():
    N = 15
    M = 10
    A = np.random.normal(0, 1, (N, M))

    opA = util.BasicLinOp(A)


    B = []

    K = 4
    m_b = 20
    V = np.zeros((M, m_b))

    V[:, 0] = np.random.normal(0, 1, M)

    W = np.zeros((N, m_b))
    F = np.zeros((N,1))
    tol = 1e-6
    sqrteps = np.sqrt(1e-14)

    SVTol = min(sqrteps, tol)

    interchange = False

    V, W, F, B, mprod = lanczos.ablanzbd(opA, V, W, F, B, K, 
                                         interchange, m_b, N, M, 
                                         SVTol, two_reorth=False, iteration=0)
    
    
    # compare the SVDs for sanity checking
    U, s, V = np.linalg.svd(A)

    U_l, s_l, V_l = np.linalg.svd(B.T)
    print s
    print s_l

    percent_diff = util.min_dist_in_frac(s_l, s)
    print percent_diff
