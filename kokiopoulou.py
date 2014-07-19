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


def bulgechasing_golub_kahan_svd(Tl, mu):
    """
    Bulgechasing (?) Golub-Kahan SVD step (algorithm 2)
    We need to implement the ability to perform p implicit QR 
    """
    y = T[0, 0]
    z = T[0, 1]
    for i in range(l-1):
        # who KNOWS what's happening here? 
        pass


def irlanb(A, k, p, eignum, tol, u_init, shifts = 'ritz'):

    l = k + p
    # compute bases U_lp1, Vl, Bl, via LBD

    while convergence == False:

        # perfom the shifts
        if shifts == 'ritz':
            U, s, V= np.linalg.svd(Bl)
        elif shifts == 'harmonic':
            pass
        
        # use bulgechasing on Bl with the p largest sigma_i as shifts
        # and update the LBD factorization 
        
        # compute approximation of min singular value
        
        # compute refined residual 
        
        

        # check convergence of singular values
