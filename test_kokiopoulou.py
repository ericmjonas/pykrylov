import numpy as np

import kokiopoulou as kok
import util

def test_k_lanczos():
    N = 1000
    A = np.random.normal(0, 1, (N, N))
    
    U, s, V = np.linalg.svd(A)

    opA = util.BasicLinOp(A)
    x_init = np.random.normal(0, 1, N)

    B, U, V = kok.lanczos_bidiagonalization(opA, x_init, 50)

    U_l, s_l, V_l = np.linalg.svd(B)
    s_l_big_to_small = np.sort(s_l)[::-1]
    percent_diff = util.min_dist_in_frac(s_l_big_to_small, s)*100

    # check for < 1% error
    np.testing.assert_array_less(np.sort(percent_diff)[:10], np.ones(10))

