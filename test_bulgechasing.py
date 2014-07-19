import numpy as np
from nose.tools import * 
import bulgechasing

def test_bc():
    N = 6
    np.random.seed(0)
    B = np.arange(N*N) + 1
    B = np.random.permutation(B)
    B.shape = N, N

    B = np.triu(B, -1)
    print B

    bulgechasing.bulgechasing_gk_svd_step(B, 0.0)
    
