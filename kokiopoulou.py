"""
Compute the smallest singular triplets 

From 

Computing smallest singular triplets with implicitly restarted
Lanczos bidiagonalization

E. Kokiopoulou, C. Bekas, E. Gallopoulos, 
Applied Numerical Mathematics 49 (2004) 39-61
doi:10.1016/j.apnum.2003.11.011

"""

import numpy as np

from util import norm


def lanczos_bidiagonalization(A, p0, k):
    """
    Lanczos bidiagonalization as presented in above paper
    (algorithm 1)

    inputs:

    Linear Operator : as defined in util
    p0 : start vector 
    k : number of iters

    output: 
    B_k :  bidiagonal real matrix, size (k+1) x k
    U_kp1  : Orthogonal bases U_{k+1} \in C, m x (k+1) 
    V_k  : Orthogonal bases V_{k} \in C, n x (k) 
    
    Note that we infer the correct dtype based on p0

    FIXME: Don't return dense B_k

    """
    
    dtype = p0.dtype

    betas = np.zeros(k+1)
    alphas = np.zeros(k)
    m, n = A.dims()
    U = np.zeros((m, k+1), dtype=dtype)
    V = np.zeros((n, k+1), dtype=dtype)

    # init
    betas[0] = norm(p0)
    U[:, 0] = p0/betas[0]
    V[:, 0] = 0 
    for i in range(k):
        r = A.compute_ATx(U[:, i]) 
        if i > 0: 
            r -= np.dot(betas[0], V[:, i]) 
        alphas[i] = norm(r)
        V[:, i+1]  = r/alphas[i]
        p = A.compute_Ax(V[:, i+1]) - alphas[i]*U[:, i]
        betas[i+1] = norm(p)
        U[:, i+1] = p / betas[i+1]

    B = np.zeros((k+1, k), dtype = dtype)

    for i in range(k):
        B[i, i] = alphas[i]
        B[i+1, i] = betas[i+1]

    return B, U, V[:, 1:]

        
    
    
    
